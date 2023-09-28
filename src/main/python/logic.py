from base import context
from PyQt5 import QtWidgets
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import pipeline
from PyPDF2 import PdfReader
import nltk


billsum = load_dataset("billsum", split="ca_test")


def openFile(window):
    fileName, _ = QtWidgets.QFileDialog.getOpenFileName(window, 'Open File', ' ', 'All Files (*);;Text Files (*.txt)')
    if fileName:
        QtWidgets.QLineEdit.setText(window.lineEdit, fileName)
    return -1

def textExtract(window):
    textlist = []
    reader = PdfReader(window.lineEdit.text())
    number_of_pages = len(reader.pages)
    complete = 0
    for page_number in range(number_of_pages):   # use xrange in Py2
        page = reader.pages[page_number]
        page_content = page.extract_text()
        #"summarize: "+page_content or page content for the nlkt version of summarize
        textlist.append("summarize: "+page_content)
        complete+=1
        window.progressBar.setValue(complete)
    window.progressBar.setValue(100)
    window.summarizeList = textlist
    print('donezo')


def summary(window):
    #always in the function
    text = window.summarizeList
    
    summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
    percent = 0
    for t in text:
        summarizer(t)

        tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
        #inputs = tokenizer.batch_encode_plus(t, max_length=1024, return_tensors="pt", pad_to_max_length=True).input_ids  # Batch size 1

        inputs = tokenizer(t, max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True).input_ids

        model = AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
        outputs = model.generate(inputs, max_new_tokens=1024, do_sample=False)

        window.textBrowser.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        percent+=1
        window.mlSummarizationStatusBar.setValue(percent)
    
    window.mlSummarizationStatusBar.setValue(100)
    print('donezo')


#FUCKING PAINFUL

    # text = window.summarizeList
    # nltk.download('punkt')
    # checkpoint = "google/pegasus-xsum"
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    # sentonce=''.join(text)
    # sentences = nltk.tokenize.sent_tokenize(sentonce)

    # # initialize
    # length = 0
    # chunk = ""
    # chunks = []
    # count = -1
    # percent = 0
    # for sentence in text:
    #     sentence = nltk.tokenize.sent_tokenize(sentence)
    #     count += 1
    #     combined_length = len(tokenizer.tokenize(sentence)) + length # add the no. of sentence tokens to the length counter

    #     if combined_length  <= tokenizer.max_len_single_sentence: # if it doesn't exceed
    #         chunk += sentence + " " # add the sentence to the chunk
    #         length = combined_length # update the length counter

    #         # if it is the last sentence
    #         if count == len(sentences) - 1:
    #             chunks.append(chunk) # save the chunk
            
    #     else: 
    #         chunks.append(chunk) # save the chunk
    #         # reset 
    #         length = 0 
    #         chunk = ""

    #         # take care of the overflow sentence
    #         chunk += sentence + " "
    #         length = len(tokenizer.tokenize(sentence))

    #     # inputs
    #     inputs = [tokenizer(chunk, return_tensors="pt") for chunk in chunks]

    #     # print summary
    #     for input in inputs:
    #         output = model.generate(**input)
    #         window.textBrowser.append(tokenizer.decode(*output, skip_special_tokens=True))
    #     percent+=1
    #     window.mlSummarizationStatusBar.setValue(percent)

    # window.mlSummarizationStatusBar.setValue(100)
    # print('donezo')