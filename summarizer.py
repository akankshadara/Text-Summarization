import re, pdb, sys, math
from collections import defaultdict
import nltk.data
import math
from nltk import word_tokenize,sent_tokenize
from collections import OrderedDict
import collections
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

# create a graph
class Graph:
	def __init__(self):
		self.Vertices = []
		self.Edges = []

	def getRankedVertices(self):
		res = defaultdict(float)
		for e in self.Edges:
			res[e.Vertex1] += e.Weight
		return sorted(res.items(), key=lambda x: x[1], reverse=True)

class Vertex:
	def __init__(self):
		self.Sentence = None

class Edge:
	def __init__(self):
		self.Vertex1 = None
		self.Vertex2 = None
		self.Weight = 0

class WordType:
	Content=0
	Function=1
	ContentPunctuation=2
	FunctionPunctuation=3

class Word:
	def __init__(self):
		self.Text=''
		self.Type=''

class Sentence:
	def __init__(self):
		self.Words = []

	def getFullSentence(self):
		text = ''
		for w in self.Words:
			text += w.Text
		return text.strip()

	def getReducedSentence(self):
		sentenceText = ''
		sentenceEnd = self.Words[len(self.Words)-1]
		contentWords = filter(lambda w: w.Type == WordType.Content, self.Words)
		i = 0
		while i < len(contentWords):
			w = contentWords[i]
			# upper case the first character of the sentence
			if i == 0:
				li = list(w.Text)
				li[0] = li[0].upper()
				w.Text = ''.join(li)
			sentenceText += w.Text
			if i < len(contentWords)-1:
				sentenceText += ' '
			elif sentenceEnd.Text != w.Text:
				sentenceText += sentenceEnd.Text
			i = i+1
		return sentenceText
			

class Paragraph:
	def __init__(self):
		self.Sentences = []

class Reduction:

	functionPunctuation = ' ,-'
	contentPunctuation = '.?!\n'
	punctuationCharacters = functionPunctuation+contentPunctuation
	sentenceEndCharacters = '.?!'
	
	def isContentPunctuation(self, text):
		for c in self.contentPunctuation:
			if text.lower() == c.lower():
				return True
		return False

	def isFunctionPunctuation(self, text):
		for c in self.functionPunctuation:
			if text.lower() == c.lower():
				return True
		return False

	def isFunction(self, text, stopWords):
		for w in stopWords:
			if text.lower() == w.lower():
				return True
		return False

	def tag(self, sampleWords, stopWords):
		taggedWords = []
		for w in sampleWords:
			tw = Word()
			tw.Text = w
			if self.isContentPunctuation(w):
				tw.Type = WordType.ContentPunctuation
			elif self.isFunctionPunctuation(w):
				tw.Type = WordType.FunctionPunctuation
			elif self.isFunction(w, stopWords):
				tw.Type = WordType.Function
			else:
				tw.Type = WordType.Content
			taggedWords.append(tw)
		return taggedWords

	def tokenize(self, text):
		return filter(lambda w: w != '', re.split('([{0}])'.format(self.punctuationCharacters), text))	

	def getWords(self, sentenceText, stopWords):
		return self.tag(self.tokenize(sentenceText), stopWords) 

	def getSentences(self, line, stopWords):
		sentences = []
		sentenceTexts = filter(lambda w: w.strip() != '', re.split('[{0}]'.format(self.sentenceEndCharacters), line))	
		sentenceEnds = re.findall('[{0}]'.format(self.sentenceEndCharacters), line)
		sentenceEnds.reverse()
		for t in sentenceTexts:
			if len(sentenceEnds) > 0:
				t += sentenceEnds.pop()
			sentence = Sentence()
			sentence.Words = self.getWords(t, stopWords)
			sentences.append(sentence)
		return sentences

	def getParagraphs(self, lines, stopWords):
		paragraphs = []
		for line in lines:
			paragraph = Paragraph()
			paragraph.Sentences = self.getSentences(line, stopWords)
			paragraphs.append(paragraph)
		return paragraphs

	def findWeight(self, sentence1, sentence2):
		length1 = len(filter(lambda w: w.Type == WordType.Content, sentence1.Words))
		length2 = len(filter(lambda w: w.Type == WordType.Content, sentence2.Words))
		if length1 < 4 or length2 < 4:
			return 0
		weight = 0
		for w1 in filter(lambda w: w.Type == WordType.Content, sentence1.Words):
			for w2 in filter(lambda w: w.Type == WordType.Content, sentence2.Words):
				if w1.Text.lower() == w2.Text.lower():
					weight = weight + 1
		normalised1 = 0
		if length1 > 0:
			normalised1 = math.log(length1)
		normalised2 = 0
		if length2 > 0:
			normalised2 = math.log(length2)
		norm = normalised1 + normalised2
		if norm == 0:
			return 0
		return weight / float(norm)

	def buildGraph(self, sentences):
		g = Graph()
		for s in sentences:
			v = Vertex()
			v.Sentence = s
			g.Vertices.append(v)
		for i in g.Vertices:
			for j in g.Vertices:
				if i != j:
					w = self.findWeight(i.Sentence, j.Sentence)
					e = Edge()
					e.Vertex1 = i
					e.Vertex2 = j
					e.Weight = w
					g.Edges.append(e)
		return g

	def sentenceRank(self, paragraphs):
		sentences = []
		for p in paragraphs:
			for s in p.Sentences:
				sentences.append(s)
		g = self.buildGraph(sentences)
		return g.getRankedVertices()


	def reduce(self, text, summary_length, GRAPH_SCORE_FACTOR):

		stopWordsFile = "stop_words.txt"
		stopWords= open(stopWordsFile).read().splitlines()

		lines = text.splitlines()
		contentLines = filter(lambda w: w.strip() != '', lines)

		paragraphs = self.getParagraphs(contentLines, stopWords)

		rankedSentences = self.sentenceRank(paragraphs)

		orderedSentences = []
		for p in paragraphs:
			for s in p.Sentences:
				orderedSentences.append(s)

		reducedSentences = []
		i = 0
		while i < summary_length:
			s = rankedSentences[i][0].Sentence
			position = orderedSentences.index(s)
			# print(s,position)
			reducedSentences.append((s, position))
			i = i + 1
		
		reducedSentences = sorted(reducedSentences, key=lambda x: x[1])
		score_from_graph_summary_dict = {}
		score = len(reducedSentences)*GRAPH_SCORE_FACTOR
		# print score
		reducedText = []
		for s,r in reducedSentences:
			sentence = s.getFullSentence()
			score_from_graph_summary_dict[sentence] = score 
			score = score - GRAPH_SCORE_FACTOR
			reducedText.append(s.getFullSentence())
		# print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
		# print reducedText
		return score_from_graph_summary_dict
	


#################################################################################################################################################
# parser - returns tokenized sentences and keywords(cleaned)
#################################################################################################################################################

# paragraph = ["Mr. Green killed Colonel Mustard in the study with the candlestick. Mr. Green is not a very nice fellow. Professor Plum has a green plant in his study. Miss Scarlett watered Professor Plum's green plant while he was away from his office last week."]


# print words_from_audio_dict
class Summarizer:

	def get_audio_word_dict(self, AMPLITUDE_FACTOR):
		global words_from_audio_dict

		audio_word_list = [word for line in open("words_from_audio.txt", 'r') for word in line.split(",")]
		words_from_audio_dict = {}

		for w in audio_word_list:
			words_from_audio_dict[w] = AMPLITUDE_FACTOR

		return words_from_audio_dict
	
	def read_from_file(self, fname):
		with open(fname) as f:
			content = f.readlines()
		f.close()
		return content

	def sentence_tokenizer(self, para):
		#use tokenizer to tokenize sentences in the sentence
		# sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
		# actual_sentences = sent_detector.tokenize(paragraph[0])
		tokenized_sentences = []
		for sentence in para:
			tokenized_sentences.append(sent_tokenize(sentence))

		tokenized_sentences = [item for sublist in tokenized_sentences for item in sublist]
		# print tokenized_sentences
		return tokenized_sentences

	def remove_punctuations(self, text_string):
		#defining a list of special characters to be used for text cleaning
		special_characters = [",",".","'",";","\n", "?", "!", ":", ")", "("] 
		cleaned_string = str(text_string)
		# removing stop words
		for ch in special_characters:
			cleaned_string = cleaned_string.replace(ch, "")
			cleaned_string = cleaned_string.lower()
		return cleaned_string

	def remove_stop_words(self, document):
		# defining a stop word list
		document = document[0].split(" ")
		# print document
		words_file = "stop_words.txt"
		stop_word_list = []
		stop_word_list = [word for line in open(words_file, 'r') for word in line.split(",")]
		cleaned_doc = []

		for term in document:
			term = self.remove_punctuations(term)
			if term not in stop_word_list:
				# print term
				cleaned_doc.append(term)
		# print cleaned_doc
		return cleaned_doc     

	def word_tokenizer(self, para):
		# removing stop words and punctuations
		clean_para = self.remove_stop_words(para)
		# print clean_para
		tokenized_words = []
		for w in clean_para:
			tokenized_words.append(word_tokenize(w))

		tokenized_words = [item for sublist in tokenized_words for item in sublist]
		# print tokenized_words
		return tokenized_words

	# ranked_sentences = sentence_ranker(tokenized_sentences)

	# remove the punctuations and stop words from the paragraph and get keywords 

	#################################################################################################################################################
	# summarizer
	#################################################################################################################################################

	# term frequency - no. of times a word appers in a sentence
	def logarithmic_tf(self, term, sentence):
		count = sentence.count(term)
		if count == 0:
			return 0
		else:
			return 1 + math.log(count)
		# returns a dictionary of all words in the corpus with their logarithmic term frequencies


	def set_freq_to_words(self, para):
		words = self.word_tokenizer(para)
		word_set = set(words)
		word_freq = {}
		for w in word_set:
			count = self.logarithmic_tf(w, words)
			word_freq[w] = count

		word_freq = OrderedDict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))
		# print word_freq
		# returns a dictionary of keywords and corresponding term frequencies
		return word_freq


	def getTitleScore(self, title, sentence):
		titleWords = remove_stop_words(title)
		sentenceWords = remove_stop_words(sentence)
		matchedWords = [word for word in sentenceWords if word in titleWords]

		return len(matchedWords) / (len(title) * 1.0)

	def sentence_ranker(self, sentences, score_from_graph_summary_dict, AMPLITUDE_FACTOR, TF_FACTOR):
		# returns a dictionary of ranked sentences
		# print score_from_graph_summary_dict
		ranked_sentences = {}
		for s in sentences:
			score = 0
			# scores of top N sentences from graph based summary
			try:
				score = score + score_from_graph_summary_dict[s]
			except:
				score = score + 0

			# score from term-frequencies of keywords
			for w in self.remove_punctuations(s).split():
				try:
					score = score + (tf_dictionary[w] * TF_FACTOR)
				except:
					score = score + 0

			# score from audio-retrieved words
			words_from_audio_dict = self.get_audio_word_dict(AMPLITUDE_FACTOR)
			for w in self.remove_punctuations(s).split():
				try:
					score = score + (words_from_audio_dict[w])
					# print w
				except:
					score = score + 0

			ranked_sentences[s] = score

		ranked_sentences = OrderedDict(sorted(ranked_sentences.items(), key=lambda x: x[1], reverse=True))
		# print ranked_sentences
		return ranked_sentences

	def summarize(self, input, tokenized_sentences, length):
		top_sentences =  dict(collections.Counter(input).most_common(length))
		# print top_sentences
		summary_file = open("summary.txt", "w") 
		summary = ""
		for s in tokenized_sentences:
			try:		
				if (s in top_sentences.keys()):
					summary += s
					summary += " "
					summary_file.write(s)
					summary_file.write(" ")
			except:
				pass
		# print "summary generated in file..."
		# print summary
		summary_file.close()
		return summary
	# get top keywords

	def generate_summary(self, input_file, SUMMARY_LENGTH):
		global GRAPH_SCORE_FACTOR
		global AMPLITUDE_FACTOR
		global TF_FACTOR
		# read input from text file, get paragraph for summarization 
		paragraph = self.read_from_file(input_file)
		# get a dictionary containing term frequencies of keywords
		tf_dictionary = self.set_freq_to_words(paragraph)
		# generate tokenized sentences
		tokenized_sentences = self.sentence_tokenizer(paragraph)
		# graph based summariztion
		score_from_graph_summary_dict = Reduction().reduce(paragraph[0], SUMMARY_LENGTH, GRAPH_SCORE_FACTOR)
		# assign score to every sentence based of Tf, corresponding amplitude, and graph edges
		ranked_sentences = self.sentence_ranker(tokenized_sentences, score_from_graph_summary_dict, AMPLITUDE_FACTOR, TF_FACTOR)
		# generate summary: get top n sentences from the ranked corpus
		summary = self.summarize(ranked_sentences, tokenized_sentences, SUMMARY_LENGTH)
		return summary

	def set_factors(self, graph, audio, tf):
		global GRAPH_SCORE_FACTOR
		global AMPLITUDE_FACTOR
		global TF_FACTOR
		GRAPH_SCORE_FACTOR = graph
		AMPLITUDE_FACTOR = audio
		TF_FACTOR = tf



