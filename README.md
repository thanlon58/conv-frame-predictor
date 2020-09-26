# conv-frame-predictor
Base code for a convolutional neural network that tries to predict frames of a video based on previous frames. It's pretty bad, but if you're willing to mess around with it you can get some cool artifacts out of it.

This isn't at all a practical endeavor, just did this for creative/artistic reasons.

# WARNING
This code is super disorganized, messy, and barely has any real documentation. I just wrote it for personal use, so I'm not gonna go through and make it super neat, so reach out if you have a question.  I will regret this choice in two months when I inevitably return to this code and can't navigate it for my life, but for now I don't care.

## Requirements:
<ul>
  <li>Tensorflow 2 w/ Keras</li>
  <li>PIL</li>
  <li>Numpy</li>
</ul>

## Architecture:
The idea behind this project is to give a network a few frames (I did 5) and ask it
to predict the sixth. I did this by just making a Convolutional NN with an input
dimension of (512,512,15)-- 3 color channels for each frame of a 512x512 image.
It's not very deep, but I still got some super interesting results.
Another thing I added was residual layers/skip connections, so the model has
what's known in the industry as a "horseshoe shape". This is all in the model
definition, and is just begging to be messed around with. <br>

I chose this architecture because it sounded fun to try out, I wanted to get
more experience with prepping data through tfrecords, and because I'm trying
everything I can think of before I try to implement a recurrent model.

## Pipeline:
If you want to use this code, here's what you're gonna have to do:
<ol>
 <li>Get your dataset.</li>
 <ul><li>I used a video of dashcam footage I got from youtube. Try to get long stretches of repetetive data.</li></ul>
 <li>Get your data saved as a tfrecord</li>
 <ul><li>This is important so that we don't need to load all the frames into
 memory at once, which would kill your computer if you tried it on like a full multi-hour video.</li></ul>
 <li>Train the model, take checkpoints of the model's weights & progress</li>
 <li>Cut off training too soon because you're impatient and just want to see
 if it looks cool</li>
 <li>Mess around with the model in a jupyter notebook</li>
</ol>

# Notes:
<ul>
  <li>I was super impatient with this model and only ever ended up training it on a tiny subset of data. If you bunker down and let it rip on a whole video you might get something better</li>
  <li>Want to try implementing a PixelCNN decoder</li>
  <li>Want to try on different data domains</li>
  <li>I trained my model on images size (512x512), but since all the layers are just convolutional, you can totally run it on higher resolutions. In some cases you can get really really interesting artifacts by doing this. Definitely something to consider messing around with.</li>
</ul>

# Some results:
![alt text](https://github.com/thanlon58/conv-frame-predictor/blob/master/results/hires3.gif?raw=true)
