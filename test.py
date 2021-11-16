from transformers import pipeline

# The generator choosen depends on the RAM capacity of the PC

# generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B') # I have 16 gb of ram use up to this option
#generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')

# The text guied and first sentence
prompt = 'La automatización es una característica esencial hoy en dia'
prompt1 = 'El tiempo es limitado y debemos cuidarlo'
prompt2 = 'Un robot es un aparato'
prompt3 = 'Telegram es una plataforma que permite'
prompt4 = 'Hoy más que nunca tenemos que aprovechar los robots para'

# The text generated here
res = generator(prompt, max_length=300, do_sample=True, temperature=0.8, min_length=150)
res1 = generator(prompt1, max_length=300, do_sample=True, temperature=0.8, min_length=150)
res2 = generator(prompt2, max_length=300, do_sample=True, temperature=0.8, min_length=150)
res3 = generator(prompt3, max_length=300, do_sample=True, temperature=0.8, min_length=150)
res4 = generator(prompt4, max_length=300, do_sample=True, temperature=0.8, min_length=150)

# Saves the generated text in a txt file
with open("Output.txt", "w") as text_file:
    text_file.write(res[0]['generated_text'])
    text_file.write(res1[0]['generated_text'])
    text_file.write(res2[0]['generated_text'])
    text_file.write(res3[0]['generated_text'])
    text_file.write(res4[0]['generated_text'])
