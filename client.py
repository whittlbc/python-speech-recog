from __future__ import division
import collections
import contextlib
import functools
import signal
import google.auth
import google.auth.transport.grpc
import google.auth.transport.requests
from google.cloud.grpc.speech.v1beta1 import cloud_speech_pb2
from google.rpc import code_pb2
import grpc
import pyaudio
from six.moves import queue
from playsound import playsound


class Client:
	RATE = 16000
	CHUNK = int(RATE / 10)  # 100ms
	SECS_OVERLAP = 1
	CHANNELS = 1
	FORMAT = pyaudio.paInt16
	# The Speech API has a streaming limit of 60 seconds of audio*, so keep the
	# connection alive for that long, plus some more to give the API time to figure
	# out the transcription.
	# * https://g.co/cloud/speech/limits#content
	DEADLINE_SECS = 60 * 3 + 5
	SPEECH_SCOPE = 'https://www.googleapis.com/auth/cloud-platform'
	BOT_NAME = 'Jarvis'
	ATTENTION_SOUND_PATH = 'attention.wav'
	
	
	def __init__(self):
		self.audio_interface = pyaudio.PyAudio()
		self.listening_for_prompt = True
		self.listening_for_command = False
	
	
	def listen(self):
		service = cloud_speech_pb2.SpeechStub(self.make_channel('speech.googleapis.com', 443))
		
		# For streaming audio from the microphone, there are three threads.
		# First, a thread that collects audio data as it comes in
		with self.record_audio(self.RATE, self.CHUNK) as buff:
			# Second, a thread that sends requests with that data
			overlap_buffer = collections.deque(maxlen=self.SECS_OVERLAP * self.RATE / self.CHUNK)
			requests = self.request_stream(self._audio_data_generator(buff, overlap_buffer), self.RATE)
			# Third, a thread that listens for transcription responses
			recognize_stream = service.StreamingRecognize(
				requests, self.DEADLINE_SECS)
			
			# Exit things cleanly on interrupt
			signal.signal(signal.SIGINT, lambda *_: recognize_stream.cancel())
			
			# Now, put the transcription responses to use.
			try:
				while True:
					self.listen_print_loop(recognize_stream, buff)
					
					# Discard this stream and create a new one.
					# Note: calling .cancel() doesn't immediately raise an RpcError
					# - it only raises when the iterator's next() is requested
					recognize_stream.cancel()
					
					requests = self.request_stream(self._audio_data_generator(
						buff, overlap_buffer), self.RATE)
					# Third, a thread that listens for transcription responses
					recognize_stream = service.StreamingRecognize(
						requests, self.DEADLINE_SECS)
			
			except grpc.RpcError:
				# This happens because of the interrupt handler
				pass
	
	
	def make_channel(self, host, port):
		"""Creates a secure channel with auth credentials from the environment."""
		# Grab application default credentials from the environment
		credentials, _ = google.auth.default(scopes=[self.SPEECH_SCOPE])
		
		# Create a secure channel using the credentials.
		http_request = google.auth.transport.requests.Request()
		target = '{}:{}'.format(host, port)
		
		return google.auth.transport.grpc.secure_authorized_channel(
			credentials, http_request, target)


	@staticmethod
	def _audio_data_generator(buff, overlap_buffer):
		"""A generator that yields all available data in the given buffer.
		Args:
				buff - a Queue object, where each element is a chunk of data.
		Yields:
				A chunk of data that is the aggregate of all chunks of data in `buff`.
				The function will block until at least one data chunk is available.
		"""
		if overlap_buffer:
			yield b''.join(overlap_buffer)
			overlap_buffer.clear()
		
		while True:
			# Use a blocking get() to ensure there's at least one chunk of data.
			data = [buff.get()]
			
			# Now consume whatever other data's still buffered.
			while True:
				try:
					data.append(buff.get(block=False))
				except queue.Empty:
					break
			
			# `None` in the buffer signals that we should stop generating. Put the
			# data back into the buffer for the next generator.
			if None in data:
				data.remove(None)
				if data:
					buff.put(b''.join(data))
				break
			else:
				overlap_buffer.extend(data)
			
			yield b''.join(data)


	def _fill_buffer(self, buff, in_data, frame_count, time_info, status_flags):
		"""Continuously collect from the audio stream, into the buffer."""
		buff.put(in_data)
		return None, pyaudio.paContinue
	
	
	# [START audio_stream]
	@contextlib.contextmanager
	def record_audio(self, rate, chunk):
		"""Opens a recording stream in a context manager."""
		# Create a thread-safe buffer of audio data
		buff = queue.Queue()
		
		audio_stream = self.audio_interface.open(
			format=self.FORMAT,
			# The API currently only supports 1-channel (mono) audio
			# https://goo.gl/z757pE
			channels=1, rate=rate,
			input=True, frames_per_buffer=chunk,
			# Run the audio stream asynchronously to fill the buffer object.
			# This is necessary so that the input device's buffer doesn't overflow
			# while the calling thread makes network requests, etc.
			stream_callback=functools.partial(self._fill_buffer, buff),
		)
		
		yield buff
		
		audio_stream.stop_stream()
		audio_stream.close()
		
		# Signal the _audio_data_generator to finish
		buff.put(None)
		self.audio_interface.terminate()  # [END audio_stream]


	@staticmethod
	def request_stream(data_stream, rate, interim_results=True):
		"""Yields `StreamingRecognizeRequest`s constructed from a recording audio
		stream.
		Args:
				data_stream: A generator that yields raw audio data to send.
				rate: The sampling rate in hertz.
				interim_results: Whether to return intermediate results, before the
						transcription is finalized.
		"""
		# The initial request must contain metadata about the stream, so the
		# server knows how to interpret it.
		recognition_config = cloud_speech_pb2.RecognitionConfig(
			# There are a bunch of config options you can specify. See
			# https://goo.gl/KPZn97 for the full list.
			encoding='LINEAR16',  # raw 16-bit signed LE samples
			sample_rate=rate,  # the rate in hertz
			# See http://g.co/cloud/speech/docs/languages
			# for a list of supported languages.
			language_code='en-US',  # a BCP-47 language tag
		)
		streaming_config = cloud_speech_pb2.StreamingRecognitionConfig(
			interim_results=interim_results,
			config=recognition_config,
			single_utterance=True,
		)
		
		yield cloud_speech_pb2.StreamingRecognizeRequest(
			streaming_config=streaming_config)
		
		for data in data_stream:
			# Subsequent requests can all just have the content
			yield cloud_speech_pb2.StreamingRecognizeRequest(audio_content=data)
	
	
	def listen_print_loop(self, recognize_stream, buff):
		"""Iterates through server responses and prints them.
		The recognize_stream passed is a generator that will block until a response
		is provided by the server. When the transcription response comes, print it.
		In this case, responses are provided for interim results as well. If the
		response is an interim one, print a line feed at the end of it, to allow
		the next result to overwrite it, until the response is a final one. For the
		final one, print a newline to preserve the finalized transcription.
		"""
		for resp in recognize_stream:
			if resp.error.code != code_pb2.OK:
				raise RuntimeError('Server error: ' + resp.error.message)
			
			if not resp.results:
				if resp.endpointer_type is resp.END_OF_UTTERANCE:
					# Signal the audio generator to stop generating, and leave the
					# buffer to fill.
					buff.put(None)
				continue
			
			result = resp.results[0]
			transcript = result.alternatives[0].transcript
			
			if self.listening_for_prompt and self.getting_bots_attention(transcript):
				self.listening_for_prompt = False
			
			if self.listening_for_command and result.is_final:
				self.listening_for_command = False
				self.listening_for_prompt = True
				print 'Heard command: ' + transcript
			
			if not self.listening_for_prompt and not self.listening_for_command and result.is_final:
				self.listening_for_command = True


	def getting_bots_attention(self, text):
		return self.BOT_NAME.lower() in text.lower().split()