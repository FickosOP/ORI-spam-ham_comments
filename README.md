# ORI-spam-ham_comments
Filip Volarić - SW54/2018

Uputstvo za pokretanje:

Za pokretanje ovog rešenja potrebno je imati instaliran
  - python verzija >= 3.0 (rađeno na 3.7 provereno na 3.8)
  - pip

## Koraci 
1. Preuzeti zip sa main grane repozitorijuma ili klonirati repozitorijum na računar (git clone)
2. Pozicionirati se na korenski folder (ORI-spam-ham_comments-main odnosno ORI-spam-ham_comments ukoliko ste clone-ovali repozitorijum)

### 1. main.py

Windows:
U komandnu liniju uneti:</br>
> pip install sklearn </br>
> pip install pandas</br>

Rešenje je moguće pokrenuti tako što ćete u komandu liniju uneti:</br>

> python main.py</br>

Linux:</br>
    - U fajlu main.py na liniji 19 umesto karaktera '\\' staviti karakter '/' </br>
    - Koraci isti kao za Windows
### 2. spam_detection.py
  (podrazumevano da ste uspešno pokrenuli main.py i instalirali gore navedene biblioteke)</br>
  
  - U komandnu liniju uneti:</br>
> pip install inflector</br>
> pip install nltk
  
Rešenje je moguće pokrenuti tako što ćete u komandu liniju uneti:

> python spam_detection.py

Razlika između ova dva rešenja je što u spam_detection.py možete videti moj pokušaj implementacije: tf-idf i bag of words modela, manipulacije podataka i procesiranja teksta, matrice konfuzije i Naive Bayes klasifikatora (NaiveBayesClassifier.py).
