curl https://nlp.stanford.edu/data/glove.6B.zip -o ./data/glove.6B.zip -L -C -
curl https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json -o ./data/train-v1.1.json -L -C -
mkdir ./models
curl https://www.dropbox.com/s/x199un20y6pv7dw/MALUUBA-CROP-SET-GLOVE.zip?dl=0 -o ./models/qgen/MALUUBA-CROP-SET-GLOVE.zip -L -C -
unzip ./models/qgen/MALUUBA-CROP-SET-GLOVE.zip -d ./models
unzip ./data/glove.6B.zip -d ./data/glove.6B/
python -m nltk.downloader punkt
python -m spacy download en
# rm ./data/glove.6B.zip
rm ./models/qgen/MALUUBA-CROP-SET-GLOVE.zip
