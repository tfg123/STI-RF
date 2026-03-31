import os


data_dir = 'data/MKG-Y'
files = ['train.txt', 'test.txt', 'valid.txt']


entities = set()
relations = set()

for file in files:
    with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            entities.add(head)
            entities.add(tail)
            relations.add(relation)


entity2id = {entity: idx for idx, entity in enumerate(sorted(entities))}
relation2id = {relation: idx for idx, relation in enumerate(sorted(relations))}


with open(os.path.join(data_dir, 'entity2id.txt'), 'w', encoding='utf-8') as f:
    for entity, idx in entity2id.items():
        f.write(f'{entity} {idx}\n')


with open(os.path.join(data_dir, 'relation2id.txt'), 'w', encoding='utf-8') as f:
    for relation, idx in relation2id.items():
        f.write(f'{relation} {idx}\n')

with open(os.path.join(data_dir, 'entities.txt'), 'w', encoding='utf-8') as f:
    for idx in sorted(entity2id.values()):
        f.write(f'{idx}\n')


with open(os.path.join(data_dir, 'relations.txt'), 'w', encoding='utf-8') as f:
    for idx in sorted(relation2id.values()):
        f.write(f'{idx}\n')


for file in files:
    with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(os.path.join(data_dir, file.replace('.txt', '2id.txt')), 'w', encoding='utf-8') as f:
        for line in lines:
            head, relation, tail = line.strip().split('\t')
            head_id = entity2id[head]
            relation_id = relation2id[relation]
            tail_id = entity2id[tail]
            f.write(f'{head_id}\t{relation_id}\t{tail_id}\n')
