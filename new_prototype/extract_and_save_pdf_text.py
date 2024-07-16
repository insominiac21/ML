import PyPDF2
import json

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def save_text_to_json(text, output_path):
    data = [{"input": line} for line in text.split('\n') if line.strip()]
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

pdf_path = "C:\\Users\\Raghu\\MedAssist\\model\\training data\\The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"
output_json_path = "C:\\Users\\Raghu\MedAssist\\output\\output.json"

text = extract_text_from_pdf(pdf_path)
save_text_to_json(text, output_json_path)
