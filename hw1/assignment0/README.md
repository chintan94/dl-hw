# This is for starters

After you open the .zip file, you should be able to see Assignment0.ipynb and /pics folder, these are two we will be using for Assignment 0.

1. To open .ipynb files locally, you have to open jupyter notebook in your virtual environment:

~~~python
source activate dlenv # for mac/linux users
activate dlenv # for windows users
jupyter notebook
~~~

​	You can check Step 5 in Local Setup on E4040 website, too.

2. To open .ipynb files on GCP, you have to first upload the file to your instance:

~~~
gcloud compute scp [LOCAL_FILE_PATH] ecbm4040@yourinstance-name:~/
~~~

​	Unzip the file, you may need to install packages first:

~~~python
sudo apt-get install zip
unzip [YOUR_FILE_NAME]
~~~

​	Then the following is just like what you do locally.