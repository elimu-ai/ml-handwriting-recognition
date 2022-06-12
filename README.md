# ml-handwriting-recognition

Machine learning model for recognizing handwritten numbers

---

Find the Model in the release section.

PreTrained Model available in onnx format. TFLITE has issues with int quantization. The model on tflite version does not suppot quantization for int8. onnx-runtime can be used for model inference.

## Training Steps

1. Clone the repository
2. Create a data, logs and models directory in the root
3. Create a sub directory named nummodel inside the models and logs directory
4. Use Conda to create a new environment using the requirements.txt
5. Run the train.py inside the src/nummodel for training

### Note
The model takes 6 hours to train on Tesla T4 for 50 epochs
Model will checkpoint within the models/nummodel directory
The train script will export the model in onnx format. Which then can be converted into any format


## About the elimu.ai Community

![elimu ai-tagline](https://user-images.githubusercontent.com/15718174/54360503-e8e88980-465c-11e9-9792-32b513105cf3.png)

 * For a high-level description of the project, see https://github.com/elimu-ai/wiki#readme
 * For project milestones, see https://github.com/elimu-ai/wiki/projects
