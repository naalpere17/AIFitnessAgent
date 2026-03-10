# AIFitnessAgent

## To install all required libraries run:

```pip install -r requirements.txt```

## To install ollama, please run these commands in order if you do not have sudo access:

```mkdir -p $HOME/.local```

```curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst   | tar --zstd -x -C $HOME/.local```

``` echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc ```

``` source ~/.bashrc ```

```ollama pull gemma3:27b```

## After that, please run this command on a separate terminal while running the agent:

```ollama server```

## To train the Intensity Recomendation model (only need to run on the first time)
```python fitness_rec/train.py```

## Running ```main.py```
* For the first 5 inputs (starting at ```age``` and ending at ```calendar link```) press enter and it will enter default values
* For the squat video path, enter ```data/king_squat.mp4``` or press enter to skip.
* When it asks if you have done a workout, type ```no``` if you don't want to enter the workout information. It is ```yes``` by default
