
# # importing modules and packages 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
# import seaborn as sns 
# from sklearn.model_selection import train_test_split 
# from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from tabulate import tabulate
# from sklearn import preprocessing 
# import pickle
# import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model,Model
# from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam


def sig(x):
 return 1/(1 + np.exp(-x))

# # importing data 
df_test = pd.read_csv('/home/anlab/Downloads/ads_data/tmp/data_test_normalized_tmp.csv', delimiter=',')
df_train = pd.read_csv('/home/anlab/Downloads/ads_data/tmp/data_train_normalized_tmp.csv', delimiter=';')
# df_val = pd.read_csv('/home/anlab/Downloads/ads_data/tmp/data_validation_normalized.csv', delimiter=';')
# df = pd.read_csv('/home/anlab/Downfor idx in df["weight"]]
print(df_test.columns)
# # Get data train, test, validation
# X_train = df_train[["normalized_pyscene_score","text_score"]] #,"normalized_weight"
y_train = df_train['scene_change']#.replace({0:False, 1:True})
# X_test = df_test[["normalized_pyscene_score","text_score","normalized_weight"]]
y_test = df_test['scene_change']#.replace({0:False, 1:True})
# # X_val = df_val[["normalized_pyscene_score","text_score","normalized_weight"]]
# # y_val = df_val['scene_change']#.replace({0:False, 1:True})

# # X = df[["pyscene_score","text_score","weight"]]
# # y = df['scene_change']#.replace({0:False, 1:True})
# # print(X) 
# # print(y) 
  
# # # # creating train and test sets 
# # # X_train, X_test, y_train, y_test = train_test_split( 
# # #     X, y, test_size=0.3, random_state=42) 
# # X_train, X_test, y_train, y_test = train_test_split(X, 
# #                                                     y, 
# #                                                     test_size=0.3, 
# #                                                     random_state=42)
# # X_train, X_val, y_train, y_val = train_test_split(X_train, 
# #                                                   y_train,
# #                                                   test_size=0.2,
# #                                                   random_state=42)

# # test_indices = X_test.index
# # train_indices = X_train.index
# # val_indices = X_val.index

# # print(test_indices)
# # print(train_indices)
# # print(val_indices)

# # # Define the model
# model = Sequential([
#     Dense(30, input_dim=3, activation='relu'),
#     Dense(1, activation='linear')
# ])

# model = load_model('/home/anlab/Downloads/ads_data/tmp/scene_change_normalized.h5')
# B_Input_Hidden = model.layers[0].get_weights()[1]
# B_Output_Hidden = model.layers[1].get_weights()[1]
# print(B_Output_Hidden)
# print(B_Input_Hidden)
# # # Compile the model
# # model.compile(loss='mean_squared_error', optimizer='adam',
# #               metrics=['mean_absolute_error'])
# # history = model.fit(X_train, y_train, epochs=50,
# #                     batch_size=30, validation_data=(X_val, y_val))
# # # # # # Evaluate the model on the test set

# # test_loss = model.evaluate(X_test, y_test)
# # # model.save('/home/anlab/Downloads/ads_data/tmp/scene_change_normalized.h5')
# # # # Make predictions
# # predictions = model.predict(X_test)
# # loss = history.history['loss']
# # val_loss = history.history['val_loss']
# # mae = history.history['mean_absolute_error']
# # val_mae = history.history['val_mean_absolute_error']
# # # Plot the training and validation loss

# # # Plot the training and validation accuracy
# # plt.figure(figsize=(18, 10))
        
# # plt.subplot(2, 1, 1)
# # plt.plot(mae, label='Training MAE')
# # plt.plot(val_mae, label='Validation MAE')
# # plt.legend()
# # plt.subplot(2, 1, 2)
# # plt.plot(loss, label='Training loss')
# # plt.plot(val_loss, label='Validation loss')
# # plt.legend()
# # plt.show()
# # print("Test Loss:", test_loss)
# # # # You can now use 'test_indices' to refer back to the original DataFrame
# # # print("Indices of X_test in the original DataFrame:", len(df["weight"][test_indices]))


# # # # creating a regression model 
# # model = LinearRegression() 
# # model.fit(X_train, y_train) 
# # predictions = model.predict(X_test) 
# # # print(predictions)
# # # predictions = np.array(predictions.round(), dtype=bool)
# # print([predictions,y_test])
# # # print(y_test)
# # predictions[predictions >= 0.5] = 1
# # predictions[predictions != True] = 0
# # # # model evaluation 
# # print('mean_squared_error : ', mean_squared_error(y_test, predictions)) 
# # print('mean_absolute_error : ', mean_absolute_error(y_test, predictions)) 
# # print('F1_score : ', accuracy_score(y_test, predictions)) 
# # plt.scatter(X_test["pyscene_score"],predictions)
# # plt.show()
# # pck_file = "Pck_LR_Model.pkl"
# # with open(pck_file, 'wb') as file:  
# #     pickle.dump(model, file)
# # test_indices = X_test.index
# # train_indices = X_train.index

# # print(df_test["video_name"].shape)

# #Save results predictions
# # dict_tmp_test = pd.DataFrame({
# #         'video_name': df_test["video_name"],
# #         'frame_number': df_test["frame_number"],
# #         'pyscene_score': df_test["pyscene_score"],
# #         'text_score': df_test["text_score"],
# #         'weight':df_test["weight"],
# #         'normalized_pyscene_score': df_test["normalized_pyscene_score"],
# #         'text_score': df_test["text_score"],
# #         'normalized_weight':df_test["normalized_weight"],
# #         'scene_change':df_test["scene_change"],
# #         "predict":predictions.reshape(-1)
# #     })
# #                 # print(dict_tmp)
# # df_test = pd.DataFrame(dict_tmp_test)
# # df_test.to_csv(f'/home/anlab/Downloads/ads_data/tmp/data_test.csv',sep=';', float_format='%.5f')

# # dict_tmp_test = pd.DataFrame({
# #         'video_name': df["video_name"][test_indices],
# #         'frame_number': df["frame_number"][test_indices],
# #         'pyscene_score': df_copy["pyscene_score"][test_indices],
# #         'text_score': df["text_score"][test_indices],
# #         'weight':df_copy["weight"][test_indices],
# #         'normalized_pyscene_score': df["pyscene_score"][test_indices],
# #         'text_score': df["text_score"][test_indices],
# #         'normalized_weight':df["weight"][test_indices],
# #         'scene_change':df["scene_change"][test_indices],
# #         # "predict":predictions
# #     })
# #                 # print(dict_tmp)
# # df_test = pd.DataFrame(dict_tmp_test)
# # df_test.to_csv(f'/home/anlab/Downloads/ads_data/tmp/data_test_normalized_tmp.csv',sep=';', float_format='%.5f')

# # dict_tmp_val = pd.DataFrame({
# #         'video_name': df["video_name"][val_indices],
# #         'frame_number': df["frame_number"][val_indices],
# #         'pyscene_score': df_copy["pyscene_score"][val_indices],
# #         'text_score': df["text_score"][val_indices],
# #         'weight':df_copy["weight"][val_indices],
# #         'normalized_pyscene_score': df["pyscene_score"][val_indices],
# #         'text_score': df["text_score"][val_indices],
# #         'normalized_weight':df["weight"][val_indices],
# #         'scene_change':df["scene_change"][val_indices],
# #         # "predict":predictions
# #     })
# #                 # print(dict_tmp)
# # df_val = pd.DataFrame(dict_tmp_val)
# # df_val.to_csv(f'/home/anlab/Downloads/ads_data/tmp/data_val_normalized_tmp.csv',sep=';', float_format='%.5f')

# # dict_tmp_train = pd.DataFrame({
# #         'video_name': df["video_name"][train_indices],
# #         'frame_number': df["frame_number"][train_indices],
# #         'pyscene_score': df_copy["pyscene_score"][train_indices],
# #         'text_score': df["text_score"][train_indices],
# #         'weight':df_copy["weight"][train_indices],
# #         'normalized_pyscene_score': df["pyscene_score"][train_indices],
# #         'text_score': df["text_score"][train_indices],
# #         'normalized_weight':df["weight"][train_indices],
# #         'scene_change':df["scene_change"][train_indices],
# #         # "predict":predictions
# #     })
# #                 # print(dict_tmp)
# # df_train = pd.DataFrame(dict_tmp_train)
# # df_train.to_csv(f'/home/anlab/Downloads/ads_data/tmp/data_train_normalized_tmp.csv',sep=';', float_format='%.5f')




# Define the inputs
input_A = Input(shape=(1,), name='input_A')  # Input for the SSIM score or similar
input_B = Input(shape=(1,), name='input_B')  # Input for the text score

# Define the neural network structure
# Neural A - learns a threshold for the visual model
neural_A = Dense(1, activation='linear', name='neural_A')(input_A)

# Neural B - learns a threshold for the text model
neural_B = Dense(1, activation='linear', name='neural_B')(input_B)

# # Neural C - learns the weighting between the two models
# concatenated = Concatenate()([neural_A, neural_B])
# neural_C = Dense(1, activation='linear', name='output_neural_C')(concatenated)

# # Define the model
# model = Model(inputs=[input_A, input_B], outputs=[neural_A, neural_B, neural_C])

# # Compile the model
# model.compile(optimizer=Adam(learning_rate=0.001),loss='mean_squared_error', metrics=['mean_absolute_error'])
# model = load_model('/home/anlab/Downloads/ads_data/tmp/scene_change_normalized_3.h5')
# # # Summary of the model
# model.summary()

# Neural C - learns the weight between the two models
concatenated = Concatenate()([neural_A, neural_B])
neural_C = Dense(1, activation='sigmoid', name='neural_C')(concatenated)  # Use sigmoid for a 0-1 output

# Define the model
model = Model(inputs=[input_A, input_B], outputs=[neural_A, neural_B, neural_C])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])
# model = load_model('/home/anlab/Downloads/ads_data/tmp/scene_change_normalized_4.h5')

# Assume you have your input data ready in `features_A` and `features_B`, and the targets in `targets`
# Train the model
history = model.fit([df_train["pyscene_score"],df_train["text_score"]], y_train, epochs=150, batch_size=16, validation_split=0.2)
model.save('/home/anlab/Downloads/ads_data/tmp/scene_change_normalized_4.h5')
# # Make predictions
predictions = model.predict([df_test["pyscene_score"],df_test["text_score"]])
# test_loss = model.evaluate([df_test["pyscene_score"],df_test["text_score"]], y_test)
print(len(predictions))
# predictions[predictions >= 0.5] = 1
# predictions[predictions != True] = 0
# # model evaluation 
# print('mean_squared_error : ', mean_squared_error(y_test, predictions)) 
# print('mean_absolute_error : ', mean_absolute_error(y_test, predictions)) 
# print('F1_score : ', accuracy_score(y_test, predictions)) 
# print(predictions[0])
output_neural_A = predictions[0] # Output of Neural A (pyscene)
output_neural_B = predictions[1]  # Output of Neural B (text)
output_neural_C = predictions[2] # Output of Neural C (combined
# output_neural_C[output_neural_C < 0.5] = 1
# output_neural_C[output_neural_C !=1] = 0
weights_neural_C = model.get_layer('neural_C').get_weights()[0]  # Weights of Neural C
# bias_neural_C = model.get_layer('output_neural_C').get_weights()[1]  # Weights of Neural C

# print(tabulate([output_neural_A, output_neural_B,output_neural_C], headers=['A', 'B', 'C']))
print(weights_neural_C)
# print(bias_neural_C)

dict_tmp_test = pd.DataFrame({
        'video_name': df_test["video_name"],
        'frame_number': df_test["frame_number"],
        'pyscene_score': df_test["pyscene_score"],
        'text_score': df_test["text_score"],
        'normalized_pyscene_score': df_test["normalized_pyscene_score"],
        'scene_change':df_test["scene_change"],
        'output_neural_A':output_neural_A.reshape(-1),
        'output_neural_B':output_neural_B.reshape(-1),
        'output_neural_C':output_neural_C.reshape(-1),
        'output_neural_A_normalized':np.array([sig(idx) for idx in predictions[0]  ]).reshape(-1),
        'output_neural_B_normalized':np.array([sig(idx) for idx in predictions[1]  ] ).reshape(-1),
        'output_neural_C_normalized':np.array([sig(idx) for idx in predictions[2]  ]  ).reshape(-1),
        "predict":output_neural_C.round().reshape(-1)
    })
                # print(dict_tmp)
df_train = pd.DataFrame(dict_tmp_test)
df_train.to_csv(f'/home/anlab/Downloads/ads_data/tmp/data_test_tmp4.csv',sep=';', float_format='%.5f')


loss = history.history['loss']
val_loss = history.history['val_loss']
# mae = history.history['mean_absolute_error']
# val_mae = history.history['val_mean_absolute_error']
neural_A_loss = history.history['neural_A_loss']
val_neural_A_loss = history.history['val_neural_A_loss']
neural_B_loss = history.history['neural_B_loss']
val_neural_B_loss = history.history['val_neural_B_loss']
neural_C_loss = history.history['neural_C_loss']
val_neural_C_loss = history.history['val_neural_C_loss']
# Plot the training and validation accuracy
plt.figure(figsize=(18, 10))
        
plt.subplot(2, 2, 1)
plt.plot(neural_A_loss, label='loss')
plt.plot(val_neural_A_loss, label='Validation')
plt.title('neural_A_loss')
plt.xlabel('Epoches')
plt.ylabel('MSE')
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(neural_B_loss, label='loss')
plt.plot(val_neural_B_loss, label='Validation')
plt.title('neural_B_loss')
plt.xlabel('Epoches')
plt.ylabel('MSE')
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(neural_C_loss, label='loss')
plt.plot(val_neural_C_loss, label='Validation')
plt.title('neural_C_loss')
plt.xlabel('Epoches')
plt.ylabel('MSE')
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.title('Train Loss')
plt.xlabel('Epoches')
plt.ylabel('MSE')
plt.legend()
plt.show()