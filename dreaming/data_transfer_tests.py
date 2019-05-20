import pickle
import numpy as np

test_dict_1 = {'message_length': 2345,
               'additional_info': 'none'}

data_1 = pickle.dumps(test_dict_1)
print(data_1)
loaded_data = pickle.loads(data_1)
print(loaded_data)

test_array = np.random.randn(2, 3)
print("test array:", test_array)
test_dict_2 = {'data': test_array.tobytes(),
               'data shape': [2, 3]}
print(test_dict_2)
data_2 = pickle.dumps(test_dict_2)
loaded_data_2 = pickle.loads(data_2)
print(loaded_data_2)
print(np.frombuffer(loaded_data_2['data']))

test_array = np.random.randn(2, 3)
test_dict_3 = {'data': test_array,
               'data shape': [2, 3]}
print(test_dict_3)
data_3 = pickle.dumps(test_dict_3)
print("data 3 length:", len(data_3))
loaded_data_3 = pickle.loads(data_3)
print(loaded_data_3)

