from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import re
import io
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import json
import subprocess

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load a new spacy model
nlp = spacy.blank("en")
db = DocBin()  # Create a DocBin object

app = Flask(__name__)

def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    return images

def extract_text_and_confidence(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    text_confidence = {}
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        if word and int(data['conf'][i]) > 0:
            text_confidence[word] = int(data['conf'][i])
    return text_confidence

def get_text_confidence_from_pdf(pdf_path):
    images = pdf_to_images(pdf_path)
    all_text_confidence = []
    for page_number, image in enumerate(images, start=1):
        text_confidence = extract_text_and_confidence(image)
        page_data = {'Page': page_number, 'Text_Confidence': text_confidence}
        all_text_confidence.append(page_data)
    return all_text_confidence

def calculate_field_confidence(df, text_confidence_data):
    field_confidence = []
    for _, row in df.iterrows():
        field = row['Field']
        data = row['Data']
        field_conf = 0
        total_words = len(data.split())
        for page_data in text_confidence_data:
            text_confidence = page_data['Text_Confidence']
            for word in data.split():
                if word in text_confidence:
                    field_conf += text_confidence[word]
        avg_confidence = field_conf / total_words if total_words > 0 else 0
        field_confidence.append(avg_confidence)
    return field_confidence

def extract_text_from_images(images):
    texts = []
    for image in images:
        with io.BytesIO() as img_buffer:
            image.save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            img = Image.open(img_buffer)
            img_text = pytesseract.image_to_string(img)
            texts.append(img_text)
    return texts

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'pdf' not in request.files:
            return redirect(request.url)
        pdf_file = request.files['pdf']
        if pdf_file.filename == '':
            return redirect(request.url)
        if pdf_file:
            pdf_path = f"./static/{pdf_file.filename}"
            pdf_file.save(pdf_path)
            images = pdf_to_images(pdf_path)
            texts = extract_text_from_images(images)
            extracted_text = "\n".join(texts)
            with open('train_data4.json') as f:
                TRAIN_DATA = json.load(f)
            cleaned_annotations = [annotation for annotation in TRAIN_DATA['annotations'] if annotation is not None]
            TRAIN_DATA['annotations'] = cleaned_annotations
            for item in tqdm(TRAIN_DATA['annotations']):
                if item is None:
                    continue
                text, annot = item
                doc = nlp.make_doc(text)
                ents = []
                for start, end, label in annot["entities"]:
                    span = doc.char_span(start, end, label=label, alignment_mode="contract")
                    if span is None:
                        print("Skipping entity")
                    else:
                        ents.append(span)
                doc.ents = ents
                db.add(doc)
            db.to_disk("./train_data4.spacy")
            
            # Initialize config file for training
            subprocess.run(["python", "-m", "spacy", "init", "config", "config.cfg", "--lang", "en", "--pipeline", "ner", "--optimize", "efficiency"])

            # Train the model
            subprocess.run(["python", "-m", "spacy", "train", "config.cfg", "--output", "./", "--paths.train", "./train_data4.spacy", "--paths.dev", "./train_data4.spacy"])

            nlp_ner = spacy.load("model-best")
            doc = nlp_ner(extracted_text)
            entities = [(ent.label_, ent.text) for ent in doc.ents]
            df = pd.DataFrame(entities, columns=['Field', 'Data'])
            text_confidence_data = get_text_confidence_from_pdf(pdf_path)
            df['Confidence'] = calculate_field_confidence(df, text_confidence_data)
            df.to_csv('1.csv', index=False)
            return render_template('index.html', tables=[df.to_html(classes='data')], titles=df.columns.values)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0")