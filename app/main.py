import pickle
import os
import numpy as np
import tensorflow as tf
import keras
import transformers as trfs
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from mangum import Mangum


app = FastAPI()
handler = Mangum(app)

PRETRAINED_MODEL_NAME = 'bert-base-uncased'
MAX_SEQUENCE_LENGTH = 64

cwd = os.path.abspath(os.path.dirname(__file__))
model_path = "./model/balanced_multi_model.h5"
model_path = os.path.abspath(os.path.join(os.path.join(cwd, model_path)))

multi_model = tf.keras.models.load_model(model_path, custom_objects={"TFBertForSequenceClassification": trfs.TFBertForSequenceClassification})

multi_mappings = {0: 'mild', 1: 'moderate', 2: 'non-depressed', 3: 'severe'}


tokenizer = trfs.BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
def prepare_data(input_text, tokenizer):
  token = tokenizer.encode_plus(
    input_text,
    max_length=MAX_SEQUENCE_LENGTH, # set the length of the sequences
    add_special_tokens=True, # add [CLS] and [SEP] tokens
    return_attention_mask=True,
    truncation=True,
    return_token_type_ids=False, # not needed for this type of ML task
    padding='max_length', # add 0 pad tokens to the sequences less than max_length
    return_tensors='tf'
  )
  return {
      'input_ids': tf.cast(token.input_ids, tf.float64),
      'attention_mask': tf.cast(token.attention_mask, tf.float64)
  }

def NormalizeData(data):
  return (data - np.min(data)) / (np.max(data) - np.min(data))

def make_prediction(model, processed_data, encoding_mapping):
  probs = model.predict(processed_data)
  probs_normalized = np.around(NormalizeData(probs[0]),decimals=3)

  prediction = encoding_mapping[np.argmax(probs[0])]
  prediction_probs = {}
  for key, val in encoding_mapping.items():
    prediction_probs[val] = str(probs_normalized[key])

  return {"prediction_probs" : prediction_probs, "prediction" : prediction}





@app.get('/predict_depression')
def predict_depression(text: str):
  processed_data = prepare_data(text, tokenizer)
  result = make_prediction(multi_model, processed_data=processed_data, encoding_mapping=multi_mappings)

  return JSONResponse(result)



if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=9000)