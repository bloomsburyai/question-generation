mkdir ./models
mkdir ./data
mkdir ./models/qgen
mkdir ./models/lm
mkdir ./models/qanet

curl https://nlp.stanford.edu/data/glove.6B.zip -o ./data/glove.6B.zip -L -C -
curl https://raw.githubusercontent.com/tomhosking/squad-du-split/master/train-v1.1.json -o ./data/train-v1.1.json -L -C -
curl https://raw.githubusercontent.com/tomhosking/squad-du-split/master/dev-v1.1.json -o ./data/dev-v1.1.json -L -C -
curl https://raw.githubusercontent.com/tomhosking/squad-du-split/master/test-v1.1.json -o ./data/test-v1.1.json -L -C -

curl http://tomho.sk/models/RL-S2S-1544356761.zip -o ./models/qgen/RL-S2S-1544356761.zip -L -C -
curl http://tomho.sk/models/lmtest.zip -o ./models/lm/lmtest.zip -L -C -
curl http://tomho.sk/models/qanet.zip -o ./models/qanet/qanet.zip -L -C -

unzip ./models/qgen/RL-S2S-1544356761.zip -d ./models/qgen
unzip ./models/lm/lmtest.zip -d ./models/lm
unzip ./models/qanet/qanet.zip -d ./models/qanet
unzip ./data/glove.6B.zip -d ./data/glove.6B/

python -m nltk.downloader punkt
python -m spacy download en
# rm ./data/glove.6B.zip
rm ./models/qgen/RL-S2S-1544356761.zip
rm ./models/lm/lmtest.zip
rm ./models/qanet/qanet.zip
