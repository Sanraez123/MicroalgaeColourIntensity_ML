import tensorflow.keras.utils as image
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tqdm import tqdm

csv_file = 'image_feature.csv'
df = pd.read_csv(csv_file)

df.head()

df_image = []
image_width = 600
image_height = 600
image_area = image_width * image_height * 3
for i in tqdm(range(df.shape[0])):
    img = image.load_img(r'image_dataset\\'+df['img_id'][i]+'.jpg', target_size=(image_width,image_height,3))
    img = image.img_to_array(img)
    df_image.append(img)
X = np.array(df_image)
X_col = len(df.index)
X = X.reshape(X_col, image_area)
X /= 255

le = LabelEncoder()
df['stress_condition'] = le.fit_transform(df['stress_condition'])

y_regr1 = np.array(df['absorbance'])
y_regr2 = np.array(df['biomass_yield'])
y_class = np.array(df['stress_condition'])
n_class = len(np.unique(y_class[:]))

X_train, X_test, y_regr1_train, y_regr1_test, y_regr2_train, y_regr2_test, y_class_train, y_class_test = train_test_split(X, y_regr1, y_regr2, y_class, test_size=0.25, random_state=1)

visible = Input(shape=(image_area,))
hidden1 = Dense(20, activation='relu', kernel_initializer='he_normal')(visible)
hidden2 = Dense(10, activation='relu', kernel_initializer='he_normal')(hidden1)
regr1_out = Dense(1, activation='linear')(hidden2)
regr2_out = Dense(1, activation='linear')(hidden2)
class_out = Dense(n_class, activation='softmax')(hidden2)
model = Model(inputs=visible, outputs=[regr1_out, regr2_out, class_out])

model.compile(loss=['mse','sparse_categorical_crossentropy'], optimizer='adam')
image.plot_model(model, to_file='model.png', show_shapes=True)

model.fit(X_train, [y_regr1_train, y_regr2_train, y_class_train], epochs=150, batch_size=32, verbose=2)
yhat1, yhat2, yhat3= model.predict(X_test)

error1 = sklearn.metrics.mean_absolute_error(y_regr1_test, yhat1)
print('MAE (absorbance): %.3f' % error1)

error2 = sklearn.metrics.mean_absolute_error(y_regr2_test, yhat2)
print('MAE (biomass_yield): %.3f' % error2)

yhat3 = np.argmax(yhat3, axis=-1).astype('int')
acc = sklearn.metrics.accuracy_score(y_class_test, yhat3)
print('Accuracy: %.3f' % acc)