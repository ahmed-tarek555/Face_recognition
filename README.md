# Face_recognition
This is a face/ gender recognition ai model.

To use it there are two ways:

Manual: 
1. You first type the name of the user to be added under the label "Enter the name of the user". 
2. And then click on "add user" and upload a photo of the user. And you may upload multiple photos of the same user of keep adding different users.
3. Click on "Register users" which saves a vector for each user added to disk.
4. Now you can click on "Recognize" and choose a picture and the model should recognize them and their gender.

Using camera:
1. Download an app on both your phone and pc that can turn your phone into a webcam.
2. Type the camera source which in our case should be the ip address of your phone and the port number of the app followed by "/video".
   (should look like this: <ip:portNumber/video>)
3. Type the name of the user to be registered.
4. Click on "camera register" and it will capture 5 pictures of faces only.
5. Click on "camera recognize" and it will recognize whoever in the video.

-----------------------------------------------------------------------------------

How it works:

it takes a face picture and a name ,and it processes the picture through the trained neural network to produce a vector
and when it's recognizing it takes a picture and passes it though the trained neural network to produce a vector and compares it 
to the vectors it has stored in the folder "known_embeddings"
-----------------------------------------------------------------------------------
It was trained on two datasets, one for gender classification and another for face recognition, the training was alternating
between the two datasets.

