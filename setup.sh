mkdir ./models
mkdir ./models/qgen

curl https://nlp.stanford.edu/data/glove.6B.zip -o ./data/glove.6B.zip -L -C -
curl https://raw.githubusercontent.com/tomhosking/squad-du-split/master/train-v1.1.json -o ./data/train-v1.1.json -L -C -
curl https://raw.githubusercontent.com/tomhosking/squad-du-split/master/dev-v1.1.json -o ./data/dev-v1.1.json -L -C -
curl https://raw.githubusercontent.com/tomhosking/squad-du-split/master/test-v1.1.json -o ./data/test-v1.1.json -L -C -

curl https://www.dropbox.com/s/l2ne2fe4wqm0buq/RL-S2S-1544356761.zip?dl=0 -o ./models/qgen/RL-S2S-1544356761.zip -L -C -

unzip ./models/qgen/RL-S2S-1544356761.zip -d ./models/qgen
unzip ./data/glove.6B.zip -d ./data/glove.6B/

python -m nltk.downloader punkt
python -m spacy download en
# rm ./data/glove.6B.zip
rm ./models/qgen/RL-S2S-1544356761.zip
