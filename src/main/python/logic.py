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
        textlist.append("summarize: "+page_content)
        complete+=1
        window.progressBar.setValue(complete)
    window.progressBar.setValue(100)
    window.summarizeList = textlist
    print('donezo')


def summary(window):
    text = window.summarizeList
    summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
    percent = 0
    for t in text:
        summarizer(t)

        tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
        inputs = tokenizer(t, return_tensors="pt").input_ids

        model = AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
        outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)

        window.textBrowser.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        percent+=1
        window.mlSummarizationStatusBar.setValue(percent)
    
    window.mlSummarizationStatusBar.setValue(100)
    print('donezo')

