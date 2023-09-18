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

# def print_data():
#     # Getting the resource data filepath
#     filepath = context.get_resource("static_data.json")

#     # Processing the resource file
#     with open(filepath) as file:
#         data = json.load(file)
#         print(data["message"])

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
    window.summarizeList = textlist
    print('donezo')


def summary(window):
    text = window.summarizeList
    summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
    for t in text:
        summarizer(t)

        tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
        inputs = tokenizer(text, return_tensors="pt").input_ids

        model = AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
        outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)

        window.textBrowser.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print('donezo')

#text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."

# billsum = billsum.train_test_split(test_size=0.2)

# checkpoint = "t5-small"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# prefix = "summarize: "


# def preprocess_function(examples):
#     inputs = [prefix + doc for doc in examples["text"]]
#     model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

#     labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

# tokenized_billsum = billsum.map(preprocess_function, batched=True)

# data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

# rouge = evaluate.load("rouge")

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

#     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
#     result["gen_len"] = np.mean(prediction_lens)

#     return {k: round(v, 4) for k, v in result.items()}

# model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# training_args = Seq2SeqTrainingArguments(
#     output_dir="my_awesome_billsum_model",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     weight_decay=0.01,
#     save_total_limit=3,
#     num_train_epochs=4,
#     predict_with_generate=True,
#     fp16=True,
#     push_to_hub=True,
# )

# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_billsum["train"],
#     eval_dataset=tokenized_billsum["test"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

# trainer.train()

# text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
