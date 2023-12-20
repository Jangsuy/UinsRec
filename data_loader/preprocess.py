

import numpy as np




def load_atomic_file(plus_item, plus_user, user_len, train_data_path):
    hist, pos_pairs = np.empty((user_len, 50)), []
    with open(train_data_path, 'r', encoding='utf-8') as file:
        file.readline()

        for line in file:
            uid, item_seq, target_iid = line.strip().split('\t')
            target_iid = int(target_iid) + plus_item
            uid = int(uid)
            item_seq = list(map(lambda x:int(x) + plus_item, item_seq.split(' ')))
            if not sum(hist[uid]):
                hist[uid] = item_seq + [0] * (50 - len(item_seq))
                pos_pairs.append([uid + plus_user, target_iid])

            # break
    return hist, np.array(pos_pairs)
    

if __name__ == '__main__':

    # ## pretraining item_embedding
    # feat_CDs = np.fromfile('data/CDs.feat1CLS', dtype=np.float32).reshape(-1, 768)
    # feat_FOOD = np.fromfile('data/FOOD.feat1CLS', dtype=np.float32).reshape(-1, 768)
    # feat_CDs = np.concatenate(([[0] * 768], feat_CDs), axis=0)
    # feat_full = np.concatenate((feat_CDs, feat_FOOD), axis=0)
    # print(feat_full)
    # np.save('data/CF_feat.npy', feat_full)

    # ## pretraining data preprocess
    # nm_list = ['CDs', 'FOOD']
    # len_list = [94010, 115349]
    # type_list = ['train', 'valid', 'test']
    # for df_type in type_list:
    #     for df_nm, user_len in zip(nm_list, len_list):
    #         plus_item = 1 if df_nm == 'CDs' else 64440 
    #         plus_user = 0 if df_nm == 'CDs' else 94010 
    #         globals()[f'{df_nm}_{df_type}_hist'], globals()[f'{df_nm}_{df_type}_pos_pairs'] = load_atomic_file(plus_item, plus_user, user_len, f'data/{df_nm}/{df_nm}.{df_type}.inter')

    #     globals()[f'CF_{df_type}_hist'] = np.concatenate((globals()[f'CDs_{df_type}_hist'],
    #                                                       globals()[f'FOOD_{df_type}_hist']), axis=0)
    #     globals()[f'CF_{df_type}_pos_pairs'] = np.concatenate((globals()[f'CDs_{df_type}_pos_pairs'],
    #                                                       globals()[f'FOOD_{df_type}_pos_pairs']), axis=0)
        
    #     np.save(f'data/CF_{df_type}_hist.npy', globals()[f'CF_{df_type}_hist'])
    #     np.save(f'data/CF_{df_type}_pos_pairs.npy', globals()[f'CF_{df_type}_pos_pairs'])

    # ## finetuning item_embedding
    # feat_Kindle = np.fromfile('data/Kindle.feat1CLS', dtype=np.float32).reshape(-1, 768)
    # feat_Kindle = np.concatenate(([[0] * 768], feat_Kindle), axis=0)
    # print(len(feat_Kindle)) # 98112 
    # np.save('data/Kindle_feat.npy', feat_Kindle)

    # ## finetuning data preprocess
    # df_nm = 'Kindle'
    # for df_type in type_list:
    #     plus_item, plus_user, user_len = 1, 0, 138436
    #     globals()[f'{df_nm}_{df_type}_hist'], globals()[f'{df_nm}_{df_type}_pos_pairs'] = load_atomic_file(plus_item, plus_user, user_len, f'data/{df_nm}/{df_nm}.{df_type}.inter')

    #     np.save(f'data/{df_nm}_{df_type}_hist.npy', globals()[f'{df_nm}_{df_type}_hist'])
    #     np.save(f'data/{df_nm}_{df_type}_pos_pairs.npy', globals()[f'{df_nm}_{df_type}_pos_pairs'])

    Kindle_test_pos_pairs = np.load('data/Kindle_test_pos_pairs.npy')
    print(Kindle_test_pos_pairs)