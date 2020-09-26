# conv-frame-predictor
Base code for a convolutional neural network that tries to predict frames of a video based on previous frames. It's pretty bad, but if you're willing to mess around with it you can get some cool artifacts out of it. I probably won't be explaining any of what the code does in-depth right now or anytime soon, so reach out if you have a question. <br>

This isn't at all a practical endeavor, just did this for creative/artistic reasons.


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
