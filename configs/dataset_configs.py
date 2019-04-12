melodic_progressive_house_default_dict = {
    'training_set': '../data/MelodicProgressiveHouseMix_train',
    'validation_set': '../data/MelodicProgressiveHouseMix_test',
    'task_set': '../data/MelodicProgressiveHouse_test',
    'unique_steps': 1.
}

melodic_progressive_house_local_test = melodic_progressive_house_default_dict.copy()
melodic_progressive_house_local_test['training_set'] = '../MelodicProgressiveHouse_Tracks_test'
melodic_progressive_house_local_test['validation_set'] = '../MelodicProgressiveHouse_Tracks_small_test'
melodic_progressive_house_local_test['task_set'] = '../MelodicProgressiveHouse_Tracks_small_test'

melodic_progressive_house_single_files = melodic_progressive_house_default_dict.copy()
melodic_progressive_house_single_files['training_set'] = '../data/MelodicProgressiveHouse_train'
melodic_progressive_house_single_files['validation_set'] = '../data/MelodicProgressiveHouse_test'
melodic_progressive_house_single_files['task_set'] = '../data/MelodicProgressiveHouse_test'
