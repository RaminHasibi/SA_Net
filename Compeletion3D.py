from torch_geometric.datasets import shapenet
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
import os
import os.path as osp
import shutil
import torch
import h5py
import re


from multiprocessing import Pool


class Completion3D(InMemoryDataset):
    url = ('http://download.cs.stanford.edu/downloads/completion3d/'
           'shapenet16K2019.zip')

    category_ids = {
        'Airplane': '02691156',
        'Bag': '02773838',
        'Cap': '02954340',
        'Car': '02958343',
        'Chair': '03001627',
        'Earphone': '03261776',
        'Guitar': '03467517',
        'Knife': '03624134',
        'Lamp': '03636649',
        'Laptop': '03642806',
        'Motorbike': '03790512',
        'Mug': '03797390',
        'Pistol': '03948459',
        'Rocket': '04099429',
        'Skateboard': '04225987',
        'Table': '04379243',
    }

    def __init__(self, root, categories=None,
                 split='train', transform=None, pre_transform=None,
                 pre_filter=None):
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)
        self.categories = categories
        super(Completion3D, self).__init__(root, transform, pre_transform,
                                       pre_filter)
        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        elif split == 'trainval':
            path = self.processed_paths[3]

        else:
            raise ValueError((f'Split {split} found, but expected either '
                              'train, val, trainval or test'))

        self.data, self.slices = torch.load(path)



    def files_in_subdirs(self, top_dir, search_pattern):
        regex = re.compile(search_pattern)
        for path, _, files in os.walk(top_dir):
            for name in files:
                full_name = osp.join(path, name)
                if regex.search(full_name):
                    yield full_name

    @property
    def raw_file_names(self):
        return list(self.category_ids.values()) + ['train_test_split']


    # def download(self):
    #     path = download_url(self.url, self.root)
    #     extract_zip(path, self.root)
    #     os.unlink(path)
    #     shutil.rmtree(self.raw_dir)
    #     name = self.url.split('/')[-1].split('.')[0]
    #     os.rename(osp.join(self.root, name), self.raw_dir)

    @property
    def processed_file_names(self):
        cats = '_'.join([cat[:3].lower() for cat in self.categories])
        return [
            os.path.join('{}_{}.pt'.format(cats, split))
            for split in ['train']
        ]

    def load_h5(self, path):
        with h5py.File(path, 'r') as f:
            data_key = list(f.keys())[0]
            data = torch.tensor(f[data_key])
            return data


    def process_filenames(self, filenames):
        data_list = []

        print(len(filenames))
        categories_ids = [self.category_ids[cat] for cat in self.categories]
        pool = Pool(10)

        for i, data in enumerate(pool.imap(self.load_h5, filenames)):
            data_list.append(data)
            print(i)
        # for i,name in enumerate(filenames):
        #     cat = name.split(osp.sep)[-2]
        #     if cat not in categories_ids:
        #         continue
        #     data_list.append(self.load_h5(name))

        pool.close()
        pool.join()

        return data_list


    def process(self, file_names=None):
        data_list = []
        for i, split in enumerate(['train']):
            for cat in self.categories:
                print(split)
                path_gt = osp.join(self.raw_dir,'shapenet',split,'gt',self.category_ids[cat])
                file_names_gt = [f for f in self.files_in_subdirs(path_gt, '.h5')]
                data_list_gt = self.process_filenames(file_names_gt)
                path_partial = osp.join(self.raw_dir, 'shapenet', split, 'partial', self.category_ids[cat])
                file_names_partial = [f for f in self.files_in_subdirs(path_partial, '.h5')]
                data_list_partial = self.process_filenames(file_names_partial)

                for gt, par in zip(data_list_gt, data_list_partial):
                    data_list += [Data(pos=par, y=gt, category=cat)]
                torch.save(self.collate(data_list), self.processed_paths[i])


if __name__ == '__main__':
    train_dataset = Completion3D('../data/Completion3D', split='train', categories='Airplane')
    print(train_dataset)