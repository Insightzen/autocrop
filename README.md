## Use
    usage: autocrop [-h] [-p PATH] [-w WIDTH] [-H HEIGHT] [-v]

    optional arguments:
      -h, --help            show this help message and exit
      -p PATH, --path PATH  folder where images to crop are located. Default:
                            photos/
      -w WIDTH, --width WIDTH
                            width of cropped files in px. Default: 500
      -H HEIGHT, --height HEIGHT
                            height of cropped files in px. Default: 500
      -v, --version         show program's version number and exit

## Test steps
Simple! In your command line, type:
```
cd ~
git clone https://github.com/Insightzen/autocrop.git
cd autocrop
pip install --upgrade .
cp remove_duplicate_files.py PATH_PHOTOS/remove_duplicate_files.py
```
Then cd into PATH_PHOTOS 
```
python remove_duplicate_files.py
autocrop -p . -o ./output
```


Autocrop is currently being tested on:
* Python:
    - 3.6
* OS:
    - Windows
