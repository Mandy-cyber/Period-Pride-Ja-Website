from flask import Flask, render_template, request
import tensorflow as tf
from bs4 import BeautifulSoup
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__) #creates app object from file

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/ask', methods=["GET", "POST"])
def my_form_post():
    if request.method == "POST":
        userQ = request.form['userQuestion']
        maxlen = 41
        tk = Tokenizer()
        new_question = [userQ]
        tk.fit_on_texts(new_question)
        seq = tk.texts_to_sequences(new_question)
        paddedQuestion = pad_sequences(seq)
        model = tf.keras.models.load_model("categoryModel.h5")
        pred = model.predict(paddedQuestion)
        labels = ['gp', 'pg', 'mp', 'mh', 'cp', 'sx']
        genCategory = labels[np.argmax(pred)]
        print(genCategory) 
        if genCategory == "gp":
            subCategoryList = ['cyclelengths', 'discharge', 'generalperiod', 'irregularperiods']
            model = tf.keras.models.load_model("gp.h5")
            pred = model.predict(paddedQuestion)
            subCategory = subCategoryList[np.argmax(pred)]
            print(subCategory) 

        elif genCategory == "pg":
            subCategoryList = ['cyclelengths', 'discharge', 'generalperiod', 'irregularperiods', 'amipregnant']
            model = tf.keras.models.load_model("pg.h5")
            pred = model.predict(paddedQuestion)
            subCategory = subCategoryList[np.argmax(pred)]
            print(subCategory) 

        elif genCategory == "mp":
            subCategoryList = ['howtouse', 'symptoms']
            model = tf.keras.models.load_model("mp.h5")
            pred = model.predict(paddedQuestion)
            subCategory = subCategoryList[np.argmax(pred)]
            print(subCategory) 

        elif genCategory == "mh":
            subCategoryList = ['gmentalhealth', 'howtomanage']
            model = tf.keras.models.load_model("mh.h5")
            pred = model.predict(paddedQuestion)
            subCategory = subCategoryList[np.argmax(pred)]
            print(subCategory) 

        elif genCategory == "cp":
            subCategoryList = ['iud', 'morningafterpill', 'birthcontrolpill', 'gcontra', 'condoms']
            model = tf.keras.models.load_model("cp.h5")
            pred = model.predict(paddedQuestion)
            subCategory = subCategoryList[np.argmax(pred)]
            print(subCategory) 

        else:
            subCategoryList = ['basics', 'bleeding', 'orgasm', 'toysandmastandporn']
            model = tf.keras.models.load_model("sx.h5")
            pred = model.predict(paddedQuestion)
            subCategory = subCategoryList[np.argmax(pred)]
            print(subCategory) 
        return render_template(f'{genCategory}.html')

    else:
        return render_template('ask.html')
    

@app.route('/info')
def info():
    return render_template('info.html')

#---------------0INFORMATION PAGE ROUTES----------------#
@app.route('/info-generalperiod')
def info_gp():
    return render_template('gp.html')

@app.route('/info-sex')
def info_sx():
    return render_template('sx.html')

@app.route('/info-pregnancy')
def info_pg():
    return render_template('pg.html')

@app.route('/info-contraceptives')
def info_cp():
    return render_template('cp.html')

@app.route('/info-mentalhealth')
def info_mh():
    return render_template('mh.html')

@app.route('/info-menstrualproducts')
def info_mp():
    return render_template('mp.html')

#--------------------------------------------------------#

@app.route('/help_out')
def help():
    return render_template('help.html')

@app.route('/test')
def test():
    return render_template('test3.html') 


if __name__ == "__main__": #'main' is the name of the file
    app.run(debug=True)
