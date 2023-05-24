from typing import Optional
from pydantic import BaseModel
from transformers import pipeline
import cv2
import os
import numpy as np
from io import BytesIO
from PIL import Image
import base64 

class Item(BaseModel):
    image: Optional[str]
    question: str
    file_url: Optional[str]

nlp = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
)

def download_file_from_url(logger, url: str, filename: str):
    logger.info("Downloading file...")

    import requests

    response = requests.get(url)
    if response.status_code == 200:
        logger.info("Download was successful")

        with open(f"/persistent-storage/{filename}", "wb") as f:
            f.write(response.content)

        return f"/persistent-storage/{filename}"

    else:
        logger.info(response)
        raise Exception("Download failed")

def answer_from_document(image, question):
    return nlp(
        image,
        question
    )


def predict(item, run_id, logger, binaries):
    item = Item(**item)

    if not item.image and not item.file_url:
        return "image or file_url field is required."

    if item.image:
        image = Image.open(BytesIO(base64.b64decode(item.image)))
    elif item.file_url:
        image = Image.open(download_file_from_url(item.file_url, run_id))

    answer = answer_from_document(image, item.question)

    return {"result": answer}