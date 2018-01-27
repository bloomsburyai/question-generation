import json

def load_squad_dataset(dev=False):
    expected_version = '1.1'
    filename = 'train-v1.1.json' if not dev else 'dev-v1.1.json'
    with open('../data/'+filename) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Expected SQuAD v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
        return(dataset)
    
if __name__ == "__main__":
    print(load_squad_dataset(False)[0]['paragraphs'][0]['qas'][0])
