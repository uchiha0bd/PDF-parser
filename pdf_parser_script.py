import os
import torch
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from transformers import AutoTokenizer, AutoModelForTokenClassification

# --- Configuration ---
# The Windows-specific paths have been REMOVED.
# The libraries will now automatically find the tools installed on Linux.
PDF_PATH = "sample_document.pdf" 

MODEL_NAME =  "philschmid/layoutlm-funsd"
MODEL_LABELS = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
ID_TO_LABEL = {i: label for i, label in enumerate(MODEL_LABELS)}
LABEL_TO_ID = {label: i for i, label in enumerate(MODEL_LABELS)}

# --- Helper Function: Normalize Bounding Boxes ---
def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

# --- Main Processing Function ---
# THE FIX: The poppler_path parameter is removed from the function definition.
def process_pdf_and_infer(pdf_path, model_name, id2label, label2id):
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading model {model_name} for token classification...")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    print(f"Converting PDF '{pdf_path}' to images and performing OCR...")
    try:
        # THE FIX: The poppler_path argument is removed.
        # The library will now search the system PATH for the tools.
        pages = convert_from_path(pdf_path)
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return

    all_processed_data = []

    for page_num, page_img in enumerate(pages):
        print(f"\n--- Processing Page {page_num + 1} ---")
        width, height = page_img.size
        ocr_data = pytesseract.image_to_data(page_img, output_type=pytesseract.Output.DICT)
        
        words, bboxes = [], []
        for i in range(len(ocr_data['text'])):
            if ocr_data['text'][i] and ocr_data['text'][i].strip() != '' and int(ocr_data['conf'][i]) > 60:
                x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                words.append(ocr_data['text'][i])
                bboxes.append([x, y, x + w, y + h])

        if not words: continue
        print(f"Found {len(words)} words on page {page_num + 1}.")
        
        normalized_bboxes = [normalize_bbox(bbox, width, height) for bbox in bboxes]
        
        encoding = tokenizer(
            text=words,
            boxes=normalized_bboxes,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        bbox = encoding["bbox"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox)

        predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
        word_ids = encoding.word_ids()

        page_results = { "page_num": page_num + 1, "extracted_entities": [] }
        current_word_idx, current_text, current_label = -1, "", ""

        for i, word_id in enumerate(word_ids):
            if word_id is not None:
                predicted_label = model.config.id2label[predictions[i]]
                
                if word_id != current_word_idx:
                    if current_word_idx != -1 and current_label != "O":
                        page_results["extracted_entities"].append({
                            "text": current_text, "label": current_label
                        })
                    
                    current_word_idx = word_id
                    current_text = words[word_id]
                    current_label = predicted_label
                else:
                    current_text += tokenizer.decode(encoding['input_ids'][0][i]).replace("##", "")

        if current_word_idx != -1 and current_label != "O":
            page_results["extracted_entities"].append({
                "text": current_text, "label": current_label
            })
        
        all_processed_data.append(page_results)

    return all_processed_data

# --- Run the Script ---
if __name__ == "__main__":
    print("Starting PDF Parser Project...")
    
    # THE FIX: The POPPLER_PATH argument is removed from the function call.
    results = process_pdf_and_infer(PDF_PATH, MODEL_NAME, ID_TO_LABEL, LABEL_TO_ID)

    if results:
        print("\n--- Final Extraction Results ---")
        for page_data in results:
            print(f"Page {page_data['page_num']}:")
            if page_data["extracted_entities"]:
                for entity in page_data["extracted_entities"]:
                    print(f"  - Text: '{entity['text']}', Label: '{entity['label']}'")
            else:
                print("  No entities extracted.")
    print("\nScript finished.")