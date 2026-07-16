from data_utils.extend import ExtendedDataset
from datasets.retina_faf_ga import RetinaFafGaDataset, RetinaFafGaSubset, RetinaFafGaSegDataset, RetinaFafGaSegSubset
from torch.utils.data import DataLoader
from utils.attribute_hashmap import AttributeHashmap


def prepare_dataset(config: AttributeHashmap, transforms_list = [None, None, None]):
    '''
    Prepare the dataset for predicting one future timepoint from one earlier timepoint.
    '''

    # Read dataset.
    if config.dataset_name == 'retina_faf_ga':
        dataset = RetinaFafGaDataset(target_dim=config.target_dim,
                                     crop_size=config.get('crop_size', 620))
        Subset = RetinaFafGaSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    # The official eye-level split, shared with GAP-INR and every other method in the comparison.
    train_indices = dataset.predefined_split['train']
    val_indices = dataset.predefined_split['val']
    test_indices = dataset.predefined_split['test']

    transforms_aug = None
    if len(transforms_list) == 4:
        transforms_train, transforms_val, transforms_test, transforms_aug = transforms_list
    else:
        transforms_train, transforms_val, transforms_test = transforms_list

    train_set = Subset(main_dataset=dataset,
                       subset_indices=train_indices,
                       return_format='one_pair',
                       transforms=transforms_train,
                       transforms_aug=transforms_aug)
    val_set = Subset(main_dataset=dataset,
                     subset_indices=val_indices,
                     return_format='all_pairs',
                     transforms=transforms_val)
    test_set = Subset(main_dataset=dataset,
                      subset_indices=test_indices,
                      return_format='all_pairs',
                      transforms=transforms_test)

    min_sample_per_epoch = 5
    if 'max_training_samples' in config.keys():
        min_sample_per_epoch = config.max_training_samples
    desired_len = max(len(train_set), min_sample_per_epoch)
    train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

    train_set = DataLoader(dataset=train_set,
                           batch_size=1,
                           shuffle=True,
                           num_workers=config.num_workers)
    val_set = DataLoader(dataset=val_set,
                         batch_size=1,
                         shuffle=False,
                         num_workers=config.num_workers)
    test_set = DataLoader(dataset=test_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=config.num_workers)

    return train_set, val_set, test_set, dataset.num_image_channel(), dataset.max_t


def prepare_dataset_segmentation(config: AttributeHashmap, transforms_list = [None, None, None]):
    # Read dataset.
    if config.dataset_name == 'retina_faf_ga':
        dataset = RetinaFafGaSegDataset(target_dim=config.target_dim,
                                        crop_size=config.get('crop_size', 620))
        Subset = RetinaFafGaSegSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    train_indices = dataset.predefined_split['train']
    val_indices = dataset.predefined_split['val']
    test_indices = dataset.predefined_split['test']

    transforms_train, transforms_val, transforms_test = transforms_list
    train_set = Subset(main_dataset=dataset,
                       subset_indices=train_indices,
                       transforms=transforms_train)
    val_set = Subset(main_dataset=dataset,
                     subset_indices=val_indices,
                     transforms=transforms_val)
    test_set = Subset(main_dataset=dataset,
                      subset_indices=test_indices,
                      transforms=transforms_test)

    min_sample_per_epoch = 5
    if 'max_training_samples' in config.keys():
        min_sample_per_epoch = config.max_training_samples
    desired_len = int(max(len(train_set) / config.batch_size, min_sample_per_epoch))
    train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

    train_set = DataLoader(dataset=train_set,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=config.num_workers)
    val_set = DataLoader(dataset=val_set,
                         batch_size=config.batch_size,
                         shuffle=False,
                         num_workers=config.num_workers)
    test_set = DataLoader(dataset=test_set,
                          batch_size=config.batch_size,
                          shuffle=False,
                          num_workers=config.num_workers)

    return train_set, val_set, test_set, dataset.num_image_channel()
