# /usr/bin/env python
# encoding=utf8
# 把LAS标注的实体关系格式转化为Baidu-InfExt的三元组格式


import json
import codecs
import click
from tqdm import tqdm

"""
all_schames:
剂量_给药日
药物_给药日
药物_给药方式
不良反应_程度
剂量_给药方式
周期_药物
药物_剂量
联合用药
时间_药物
药物_不良反应
"""

@click.command()
@click.option('-i', 'inputfile', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('-o', 'outputfile', type=click.Path())
@click.option('-re', 'all_schemas', type=click.Path())
def trans2bdie(inputfile, outputfile, all_schemas):
    out_fd = codecs.open(outputfile, mode='w', encoding='utf8')
    rel_schames = set()
    with codecs.open(inputfile, encoding='utf8') as fd:
        jdata = json.load(fd)
        for elem in tqdm(jdata, desc='processing'):
            text = elem.get('text', '')
            jobid = elem.get('job_id', 0)
            ann = elem.get('ann', [])
            spo_list, rel_schame = ann2spo(text, ann)
            elem_new = {'jobid': jobid, 'text': text, 'spo_list': spo_list}
            out_fd.write(json.dumps(elem_new, ensure_ascii=False))
            out_fd.write('\n')
            rel_schames |= rel_schame

    with codecs.open(all_schemas, mode='w', encoding='utf8') as ofd:
        for e in rel_schames:
            ofd.write(e)
            ofd.write('\n')


def ann2spo(text, ann):
    result = []
    rel_schames = []
    words = ann.get('words', [])
    word_id_idx = word_idx(words)
    for rel in ann.get('relationList', []):
        fid, tid = rel['fromWord']['id'], rel['toWord']['id']
        rel_tag = rel['relationTag']
        elem = {'predicate': rel_tag,
                'subject_type': word_id_idx[fid]['entityTag'],
                'object_type':  word_id_idx[tid]['entityTag'],
                'subject': text[word_id_idx[fid]['start']:word_id_idx[fid]['end']],
                'object':  text[word_id_idx[tid]['start']:word_id_idx[tid]['end']] }
        result.append(elem)
        rel_schames.append(json.dumps({"subject_type": word_id_idx[fid]['entityTag'],
                            "predicate": rel_tag,
                            "object_type": word_id_idx[tid]['entityTag'],
                            }, ensure_ascii=False))
    return result, set(rel_schames)


def word_idx(words):
    """
    :param words: [{id:x, start:x, end:x, entityTag:x}, ]
    :return:
    """
    result = {}
    for wd in words:
        id = wd.get('id', None)
        if not id:
            raise Exception('id not in words')
        if id not in result:
            result[id] = wd
    return result


if __name__ == '__main__':
    trans2bdie()
