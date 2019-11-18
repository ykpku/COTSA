import os
import sys
import re
from collections import defaultdict
from collections import Counter

from Utils.utils_siamese import paint_xy

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,.)(; +-_:/!&?|\\\'~*–#−'


def is_valid(ss):
    result = False
    vv = "abcdefghijklmnopqrstuvwxyz0123456789"
    for s in ss:
        if s in vv:
            result = True
            break
    return result


def get_char_dict(file_name="E:/githubWorkSpace/KnowledgeAlignmentCode/knowledgeAlignment_merged/dataset/dbp_wd/mapping/0_3/attr/attr_triples_1"):
    dp_att_trip_file = file_name
    ent_ids, att_ids, values = [], [], []
    with open(dp_att_trip_file, 'r', encoding="utf-8") as f:
        att_val_count = Counter()
        num, line_num = 0, 0
        length_count = defaultdict(int)
        for line_num, trip in enumerate(f.readlines()):
            att_sp = trip.split("\t")
            att_val = att_sp[2].strip()
            if att_val.endswith("@en"):
                att_val = att_val[:-3]
            print(att_val)
            if "\\U" not in att_val and "\\N" not in att_val:
                att_val = eval("u'" + (att_val.strip().lstrip("\"").rstrip("\"")).strip("\\").replace("'", "\\'") + "'")
            chars = [ch for ch in att_val if ch in alphabet]
            filtered = ''.join(chars).lower().strip()
            # print(filtered)
            if len(filtered) == len(att_val):
                print(att_val, "----", filtered)
                filtered = filtered.replace('−', '-')
                filtered = filtered.replace('–', '-')
                filtered = filtered.replace('\\', '')
                filtered = filtered.replace('*', '')
                filtered = filtered.replace('~', '')
                filtered = filtered.replace('#', '')
                filtered = filtered.replace('+', '')
                # print(att_val)
            if not is_valid(filtered):
                # print(filtered)
                pass
            else:
                ent_ids.append(att_sp[0])
                att_ids.append(att_sp[1])
                values.append(filtered)
                length_count[len(filtered)] += 1
                att_val_count.update(filtered)
                num += 1
    print(att_val_count.most_common(20))
    print(line_num, num)
    # write_counter_2file(att_val_count, os.path.dirname(curPath) + "\dataset\dp_att_value_count.csv")
    print(length_count)
    x, y = [], []
    keys = length_count.keys()
    keys = sorted(keys)
    # print(keys)
    for k in keys:
        last = 0
        if len(x) > 0:
            last = y[len(x) - 1]
        # print('k', last)
        x.append(k)
        y.append(length_count[k] + last)
    paint_xy(x, y)
    print(len(set(att_ids)))
    print(len(set(ent_ids)))
    print(len(set(values)))
    return ent_ids, att_ids, values


def fileted_attr(tri_name="attr_triples_1", dp_att_file="E:/githubWorkSpace/KnowledgeAlignmentCode/knowledgeAlignment_merged/dataset/dbp_wd/mapping/0_3/attr/"):
    trip_ent_ids, trip_att_ids, trip_value = get_char_dict(file_name=tri_name)
    with open(dp_att_file + "filtered_wiki_attr_triples", "w", encoding="utf-8") as f:
        for i in range(len(trip_ent_ids)):
            f.write(trip_ent_ids[i] + "\t" + trip_att_ids[i] + "\t" + trip_value[i] + "\n")


def test_clean_attr1_rules(att_folder, attr_id_name="attr_ids_1", attr_type_name="attr_type_1", attr_trip_name="filtered_attr_triples_1"):
    # delete the noise of attr1

    with open(att_folder + attr_id_name, 'r', encoding='utf8') as f:
        with open(att_folder + "clean_new_att_ids_1", 'w', encoding="utf-8")as wf:
            attr_name_delete = []
            attr_name_use = {}
            old_attr_ids = []
            for line in f.readlines():
                params = line.strip().split('\t')
                if len(params) != 2:
                    print("wrong line" + line)

                name = params[0].strip()[1:]
                old_attr_ids.append(name)
                if name in ['http://dbpedia.org/ontology/status',
                            'http://dbpedia.org/ontology/latestReleaseVersion', 'http://dbpedia.org/ontology/latestPreviewVersion', 'http://dbpedia.org/ontology/areaCode', 'http://dbpedia.org/ontology/utcOffset', 'http://dbpedia.org/ontology/leaderTitle', 'http://dbpedia.org/ontology/areaWater', 'http://dbpedia.org/ontology/postalCode', 'http://dbpedia.org/ontology/populationAsOf', 'http://dbpedia.org/ontology/numberOfPages', 'http://dbpedia.org/ontology/areaLand', 'http://dbpedia.org/ontology/areaWater', 'http://dbpedia.org/ontology/percentageOfAreaWater', 'http://dbpedia.org/ontology/populationDensity', 'http://dbpedia.org/ontology/censusYear', 'http://dbpedia.org/ontology/orderInOffice', 'http://dbpedia.org/ontology/serviceEndYear', 'http://dbpedia.org/ontology/serviceStartYear', 'http://dbpedia.org/ontology/militaryCommand', 'http://dbpedia.org/ontology/signature', 'http://dbpedia.org/ontology/areaWater', 'http://dbpedia.org/ontology/productionStartYear', 'http://dbpedia.org/ontology/productionEndYear', 'http://dbpedia.org/ontology/wheelbase', 'http://dbpedia.org/ontology/transmission']:
                    attr_name_delete.append(name)
                else:
                    attr_name_use[name] = len(attr_name_use)
                    wf.write(name + "\t" + str(attr_name_use[name]) + "\n")
        print("deleted attr size:", len(attr_name_delete), "attr with use size:", len(attr_name_use), "==", len(old_attr_ids))
        print(attr_name_delete)

    with open(att_folder + attr_type_name, 'r', encoding='utf8') as f:
        with open(att_folder + 'clean_new_att_type_1', 'w', encoding="utf-8")as wf:
            for line in f.readlines():
                params = line.strip().split('\t')
                if len(params) != 2:
                    print("wrong line: " + line)
                    continue
                h = int(params[0])
                a = params[1]
                if old_attr_ids[h] in attr_name_use:
                    wf.write(str(attr_name_use[old_attr_ids[h]]) + "\t" + a + "\n")

    with open(att_folder + attr_trip_name, 'r', encoding='utf8') as f:
        old_ent = defaultdict(lambda: 0)
        new_ent = defaultdict(lambda: 0)
        with open(att_folder + 'clean_new_filtered_att_triples_1', 'w', encoding="utf-8")as wf:
            for line in f.readlines():
                params = line.strip().split('\t')
                if len(params) != 3:
                    print("wrong line" + line)
                h = params[0]
                a = int(params[1])
                v = params[2]
                old_ent[h] += 1
                if old_attr_ids[a] in attr_name_use:
                    new_ent[h] += 1
                    if old_attr_ids[a] == 'http://dbpedia.org/ontology/isoCodeRegion':
                        v = v[:2]
                    wf.write(h + "\t" + str(attr_name_use[old_attr_ids[a]]) + "\t" + v + "\n")
        print(len(old_ent), "==>", len(new_ent))


def _clean_kb2_att(att_folder, attr_id_name="wiki_attr_ids", attr_type_name="wiki_attr_type", attr_trip_name="filtered_wiki_attr_triples", append="clean_", new_attr_ids={}):
    old_attr_ids = []
    with open(att_folder + attr_id_name, 'r', encoding='utf8') as f:
        for line in f.readlines():
            params = line.strip().split('\t')
            if len(params) != 2:
                print("wrong line" + line)
            h = params[0][1:].strip()
            old_attr_ids.append(h)

    with open(att_folder + attr_type_name, 'r', encoding='utf8') as f:
        with open(att_folder + append + attr_type_name, 'w', encoding="utf-8")as wf:
            for line in f.readlines():
                params = line.strip().split('\t')
                if len(params) != 2:
                    print("wrong line" + line)
                    continue
                h = int(params[0])
                a = params[1]
                if old_attr_ids[h] in new_attr_ids:
                    wf.write(str(new_attr_ids[old_attr_ids[h]]) + "\t" + a + "\n")

    with open(att_folder + attr_trip_name, 'r', encoding='utf8') as f:
        old_ent = defaultdict(lambda: 0)
        new_ent = defaultdict(lambda: 0)
        with open(att_folder + append + attr_trip_name, 'w', encoding="utf-8")as wf:
            for line in f.readlines():
                params = line.strip().split('\t')
                if len(params) != 3:
                    print("wrong line" + line)
                h = params[0]
                a = int(params[1])
                v = params[2]
                old_ent[h] += 1
                if old_attr_ids[a] in new_attr_ids:
                    new_ent[h] += 1
                    if old_attr_ids[a] in ['inception', 'time_of_earliest_written_record', 'start_time']:
                        v = v[:-10]
                    if old_attr_ids[a] == 'FIPS_10-4_(countries_and_regions)':
                        v = v[:2]
                    wf.write(h + "\t" + str(new_attr_ids[old_attr_ids[a]]) + "\t" + v + "\n")
        print(len(old_ent), "==>", len(new_ent))


def test_clean_attr2_rules(attr_path, origin_name_file="E:/githubWorkSpace/KnowledgeAlignment/dataset/0_3/attr/wiki_attr_name"):
    wiki_att_dict = {}
    with open(origin_name_file, 'r', encoding='utf8')as f:
        for line in f.readlines():
            sp = line.split("\t")
            if len(sp) != 2:
                print(sp)
            name = sp[0].strip()
            wiki_att_dict[name] = sp[1].strip().replace(" ", "_")
    # delete the noise of attr2
    with open(attr_path + "wiki_attr_ids", 'r', encoding='utf8') as f:
        with open(attr_path + "clean_new_att_ids_2", 'w', encoding="utf-8")as wf:
            attr_url_use = {}
            attr_name_ID = []
            attr_name_use = []
            for line in f.readlines():
                params = line.strip().split('\t')
                if len(params) != 2:
                    print("wrong line" + line)

                att_url = params[0][1:].strip()
                if att_url in wiki_att_dict:
                    name = wiki_att_dict[att_url]
                else:
                    name = att_url
                if name.endswith("_ID") or name in ['SourceForge_project', 'Dewey_Decimal_Classification', 'Gentoo_package', 'FIPS_55-3_(locations_in_the_US)', 'coordinates_of_northernmost_point', 'coordinates_of_southernmost_point', 'coordinates_of_easternmost_point', 'coordinates_of_westernmost_point', 'US_National_Archives_Identifier', 'UN/LOCODE', 'FIPS_6-4_(US_counties)', 'Libris-URI']:
                    attr_name_ID.append(name)
                else:
                    attr_name_use.append(name)
                    attr_url_use[att_url] = len(attr_url_use)
                    wf.write(att_url + "\t" + str(attr_url_use[att_url]) + "\n")
        print("attr with ID size:", len(attr_name_ID), "attr with use size:", len(attr_name_use), "==", len(attr_url_use))
        print(attr_name_ID[:10])
        print(attr_name_use[:10])
        _clean_kb2_att(att_folder=attr_path, new_attr_ids=attr_url_use)


def find_wiki_name(origin_name_file="E:/githubWorkSpace/KnowledgeAlignment/dataset/0_3/attr/wiki_attr_name", wiki_att_id_file="E:/githubWorkSpace/KnowledgeAlignmentDataset/15kdata/db-wiki_V2/attrs/wiki_attr_ids", out_file="E:/githubWorkSpace/KnowledgeAlignmentDataset/15kdata/db-wiki_V2/attrs/wiki_attr_names"):
    wiki_att_dict = {}
    with open(origin_name_file, 'r', encoding='utf8')as f:
        for line in f.readlines():
            sp = line.split("\t")
            if len(sp) != 2:
                print(sp)
            name = sp[0].strip()
            wiki_att_dict[name] = sp[1].strip()
    all_att = 0
    in_att = 0
    with open(out_file, 'w', encoding="utf8")as wf:
        with open(wiki_att_id_file, 'r', encoding='utf8')as f:
            for l in f.readlines():
                all_att += 1
                spl = l.split("\t")
                ur = spl[0][1:].strip()
                if ur in wiki_att_dict:
                    wf.write(ur + "\t" + wiki_att_dict[ur] + "\t" + spl[1].strip() + "\n")
                    in_att += 1
    print(in_att, "--", all_att)





# fileted_attr(tri_name="E:/githubWorkSpace/KnowledgeAlignmentDataset/15kdata/db-wiki_V2/attrs/dbp_attr_triples", dp_att_file="E:/githubWorkSpace/KnowledgeAlignmentDataset/15kdata/db-wiki_V2/attrs/")
# test_clean_attr1_rules(att_folder="E:/githubWorkSpace/KnowledgeAlignmentDataset/15kdata/db-wiki_V2/attrs/", attr_id_name="dbp_attr_ids", attr_type_name="dbp_attr_type", attr_trip_name="filtered_dbp_attr_triples")

# find_wiki_name()
# fileted_attr(tri_name="E:/githubWorkSpace/KnowledgeAlignmentDataset/15kdata/db-wiki_V2/attrs/wiki_attr_triples", dp_att_file="E:/githubWorkSpace/KnowledgeAlignmentDataset/15kdata/db-wiki_V2/attrs/")
# test_clean_attr2_rules(attr_path="E:/githubWorkSpace/KnowledgeAlignmentDataset/15kdata/db-wiki_V2/attrs/")

# step 1 : get the attr_value_length for each entity in knowledge graph
get_char_dict(file_name="E:/githubWorkSpace/KnowledgeAlignmentDataset/15kdata/db-wiki_V2/attrs/clean_filtered_wiki_attr_triples")

# dbpedia-wiki-V1 data
# att1 : attr_num==10 attr_value_length==28 (25-30) att_kind_num=343
# att2 : attr_num==18 attr_value_length==38 att_kind_num=611

# dbpedia-wiki-V2 data
# att1 : attr_num==10 attr_value_length==40 att_kind_num=236
# att2 : attr_num==18 attr_value_length==38-40 att_kind_num=435

# step 4: set the parameters in attr_param.py and param.py according to above steps. and change the input file of run.py
# step 5: run run.py

