# AIFitnessAgent

## To install all required libraries run:

```pip install -r requirements.txt```

## To install ollama, please run these commands in order if you do not have sudo access:

```mkdir -p $HOME/.local```

```curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst   | tar --zstd -x -C $HOME/.local```

``` echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc ```

``` source ~/.bashrc ```

```ollama pull gemma3:27b```
```ollama pull gpt-oss:20b```

## After that, please run this command on a separate terminal while running the agent:

```ollama serve```
Sometimes connecting to the gpt-oss model will fail with a 500 error while the code is running, in this case, ollama has timed out and needs to be restarted.
**NOTE**: make sure the ollama server is running on a separate GPU with the main.py process since the server uses a lot of vram. In order to run a process on a different GPU, an example command is: `CUDA_VISIBLE_DEVICES=0 ollama serve`

## To train the Intensity Recomendation model (only need to run on the first time)
```python fitness_rec/train.py```

## Running the main program and agent (`main.py`)
**NOTE**: like with ollama serve, make sure main is run on an unused GPU, so if ollama is on GPU 0, run main with `CUDA_VISIBLE_DEVICES=1 python3 main.py`.
* For the first 5 inputs (starting at `age` and ending at `calendar link`) enter a value or press enter and it will enter default values
* For the squat video path, enter `data/king_squat.mp4` or press enter to skip.
* When asked when you want to work out, you can respond with "today/tomorrow", "anytime this week", or similar to check the calendar for availability on those days.
* When it asks if you have done a workout, type `no` if you don't want to enter the workout information. It is `yes` by default
* The AI agent will provide a summary first, then you can respond to it and ask questions about workouts.
* When running main multiple times, you can choose to reuse the generated data from your previous run, to avoid answering all these questions again.

## To quickly check squat form on a video: 
```python check_squat_form.py data/king_squat.mp4```

## To extract frames and build the dataset from the Zenodo squat dataset: 
First, download the dataset from https://zenodo.org/records/17558630 and add it to the root directory. Then build it using the command `python -m scripts.build_zenodo_frame_dataset`

## After building the dataset, train the form detection models with: 
```python -m scripts.train_zenodo_frame_model```


