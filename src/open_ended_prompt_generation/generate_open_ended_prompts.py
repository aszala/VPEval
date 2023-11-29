import json
from tqdm import tqdm
from multiprocessing import Pool

import openai

openai.api_key = ''

prompt = """
Given a image descriptions, generate one or two programs that verifies if the image description is correct.
Classify each concept into a type (object, human, animal, food, activity, attribute, counting, color, material, spatial, location, shape, text, other), and then generate a program for each type.

Description: A man posing for a selfie in a jacket and bow tie.
Entities: man, selfie, jacket, bow tie
Activities: posing
Colors:
Counting:
Text:
Other attributes:
Questions and programs are below:
About man (human):
objectEval(image, 'man')
vqa(image, 'who is posing for a selfie?', 'man,woman,boy,girl', 'man')
About selfie (activity):
vqa(image, 'is the man taking a selfie?', 'yes,no', 'yes')
vqa(image, 'what type of photo is the person taking?', 'selfie,landscape,sports,portrait', 'selfie')
About jacket (object):
vqa(image, 'is the man wearing a jacket?', 'yes,no', 'yes')
objectEval(image, 'jacket')
About bow tie (object):
objectEval(image, 'bow tie')
vqa(image, 'is the man wearing a bow tie?', 'yes,no', 'yes')
vqa(image, 'is the man wearing a bow tie or a neck tie?', 'bow tie,neck tie,cravat,bolo tie', 'bow tie')
About posing (activity):
vqa(image, 'is the man posing for the selfie?', 'yes,no', 'yes')
vqa(image, 'what is the man doing besides taking the selfie?', 'posing,waving,nothing,shaking', 'posing')

Description: A horse and several cows feed on hay.
Entities: horse, cows, hay
Activities: feed on
Colors:
Counting: several
Text:
Other attributes:
Questions and programs are below:
About horse (animal):
objectEval(image, 'horse')
About cows (animal):
objectEval(image, 'cows')
About hay (object):
objectEval(image, 'hay')
vqa(image, 'what are the horse and cows feeding on?', 'hay,grass,leaves,twigs', 'hay')
About feed on (activity):
vqa(image, 'are the horse and cows feeding on hay?', 'yes,no', 'yes')
About several (counting):
countEval(objDet(image, 'cows'), '>1')

Description: A red colored dog.
Entities: dog
Activities:
Colors: red
Counting:
Text:
Other attributes:
Questions and programs are below:
About dog (animal):
objectEval(image, 'dog')
vqa(image, 'what animal is in the picture?', 'dog,cat,bird,fish', 'dog')
About red (color):
vqa(image, 'is the dog red?', 'yes,no', 'yes')
vqa(image, 'what color is the dog?', 'red,black,white,yellow', 'red')

Description: A busy intersection with an ice cream truck driving by.
Entities: intersection, ice cream truck
Activities: driving by
Colors:
Counting:
Text:
Other attributes: busy
Questions and programs are below:
About intersection (location):
vqa(image, 'is this an intersection?', 'yes,no', 'yes')
vqa(image, 'is this an intersection or a straight road?', 'intersection,straight road,parking lot,school', 'intersection')
About ice cream truck (object):
objectEval(image, 'ice cream truck')
vqa(image, 'is this an ice cream truck?', 'yes,no', 'yes')
About driving by (activity):
vqa(image, 'is the ice cream truck driving by?', 'yes,no', 'yes')
vqa(image, 'is this a busy intersection?', 'yes,no', 'yes')
vqa(image, 'is this a busy or a quiet intersection?', 'busy,quiet,silent,noisy', 'busy')

Description: Portrait of a gecko wearing a train conductor's hat and holding a flag that has a yin-yang symbol on it. Woodcut.
Entities: gecko, train conductor's hat, flag, yin-yang symbol
Activities: wearing, holding
Colors:
Counting:
Text:
Other attributes: portrait, woodcut
Questions and programs are below:
About gecko (animal):
objectEval(image, 'gecko')
vqa(image, 'what animal is in the photo?', 'gecko,human,snake,frog', 'gecko')
About train conductor's hat (object):
objectEval(image, 'train conductor's hat')
About flag (object):
objectEval(image, 'flag')
vqa(image, 'is the gecko holding a flag?', 'yes,no', 'yes')
vqa(image, 'what is the gecko holding?', 'flag,sign,banner,poster', 'flag')
About yin-yang symbol (object):
vqa(image, 'is there a yin-yang symbol on the flag?', 'yes,no', 'yes')
vqa(image, 'what is on the flag?', 'yin-yang symbol,star,heart,cross', 'yin-yang symbol')
About wearing (activity):
vqa(image, 'is the gecko wearing a train conductor's hat?', 'yes,no', 'yes')
About holding (activity):
vqa(image, 'is the gecko holding a flag?', 'yes,no', 'yes')
About portrait (attribute):
vqa(image, 'is this a portrait?', 'yes,no', 'yes')
About woodcut (attribute):
vqa(image, 'is this a woodcut?', 'yes,no', 'yes')
vqa(image, 'is this a woodcut, a painting, or a photograph?', 'woodcut,painting,drawing,photograph', 'woodcut')

Description: A woman is showing a watermelon slice to a woman on a scooter.
Entities: woman, watermelon slice, woman, scooter
Activities: showing
Colors:
Counting: two
Text:
Other attributes: on
Questions and programs are below:
About woman (human):
objectEval(image, 'woman')
vqa(image, 'who is showing a watermelon slice?', 'woman,man,boy,girl', 'woman')
About watermelon slice (food):
objectEval(image, 'watermelon slice')
vqa(image, 'what is the woman showing to a woman on a scooter?', 'watermelon slice,apple,orange,banana', 'watermelon slice')
About woman (human):
vqa(image, 'is there a woman on a scooter?', 'yes,no', 'yes')
vqa(image, 'who is on a scooter?', 'woman,man,boy,girl', 'woman')
About scooter (object):
objectEval(image, 'scooter')
vqa(image, 'what vehicle is one of woman on?', 'scooter,bicycle,motorcycle,car', 'scooter')
About two (counting):
countEval(objDet(image, 'woman'), '==2')
About showing (activity):
vqa(image, 'is the woman showing a watermelon slice to another woman?', 'yes,no', 'yes')
About on (spatial):
spatialEval(image, 'woman,scooter,above')
vqa(image, 'is one of the woman on a scooter?', 'yes,no', 'yes')
vqa(iamge, 'is the woman on the scooter or standing next to it?', 'on,next to,behind,in front of', 'on')

Description: A photo of three dogs.
Entities: dogs
Activities:
Colors:
Counting: three
Text:
Other attributes: photo
Questions and programs are below:
About dogs (animal):
objectEval(image, 'dog')
vqa(image, 'what animals are in the photo?', 'dogs,cats,birds,fish', 'dogs')
About three (counting):
countEval(objDet(image, 'dog'), '==3')
vqa(image, 'how many dogs are in the photo?', '1,2,3,4', '3')
About photo (attribute):
vqa(image, 'is this a photo?', 'yes,no', 'yes')
vqa(image, 'is this a photo, a painting, or a comic?', 'photo,painting,comic,sculpture', 'photo')

Description: A white milk truck with a license plate that reads \"pro milk.\"
Entities: milk truck, license plate
Activities:
Colors: white
Counting:
Text: pro milk
Other attributes:
Questions and programs are below:
About milk truck (object):
objectEval(image, 'milk truck')
About license plate (object):
objectEval(image, 'license plate')
vqa(image, 'is there a license plate on the milk truck?', 'yes,no', 'yes')
About white (color):
vqa(image, 'is the truck white?', 'yes,no', 'yes')
vqa(image, 'what color is the truck?', 'white,black,red,blue', 'white')
About pro milk (text):
textEval(ocr(image), 'pro milk')

Description: A person sitting on a horse in air over gate in grass with people and trees in background.
Entities: person, horse, gate, grass, people, trees
Activities: sitting
Colors:
Counting: several
Text:
Other attributes: over, in air, background
Questions and programs are below:
About person (human):
objectEval(image, 'person')
vqa(image, 'who is sitting on a horse?', 'person,animal,robot,alien', 'person')
About horse (animal):
objectEval(image, 'horse')
vqa(image, 'what animal is in the picture?', 'horse,cow,sheep,goat', 'horse')
About gate (object):
objectEval(image, 'gate')
About grass (object):
objectEval(image, 'grass')
About several (counting):
countEval(objDet(image, 'person'), '>1')
About trees (object):
objectEval(image, 'tree')
About sitting (activity):
vqa(image, 'is the person sitting on the horse?', 'yes,no', 'yes')
spatialEval(image, 'person,horse,above')
About over (spatial):
spatialEval(image, 'horse,gate,above')
vqa(image, 'is the horse over the gate?', 'over,under,next to,behind', 'over')
About in air (attribute):
vqa(image, 'is the horse in air?', 'yes,no', 'yes')
vqa(image, 'is the horse in air or on the ground?', 'in air,on the ground,in the water,in the sky', 'in air')
About background (attribute):
vqa(image, 'are there people and trees in the background?', 'yes,no', 'yes')
vqa(image, 'are there people and trees in the background or in the foreground?', 'background,foreground,middle ground,sky', 'background')

Description: a red blue and yellow train and some people on a platform
Entities: train, people, platform
Activities:
Colors: red blue and yellow
Counting: some
Text:
Other attributes:
Questions and programs are below:
About train (object):
objectEval(image, 'train')
vqa(image, 'is this a train?', 'train,car,motorcycle,bus', 'train')
About people (human):
objectEval(image, 'person')
About platform (location):
objectEval(image, 'platform')
vqa(image, 'what type of place is this?', 'platform,parking lot,airport,bus stop', 'platform')
About red blue and yellow (color):
vqa(image, 'is the train red blue and yellow?', 'yes,no', 'yes')
vqa(image, 'what color is the train?', 'red blue and yellow,pure blue,blue and yellow,red and blue', 'red blue and yellow')
About some (counting):
countEval(objDet(image, 'person'), '>0')

Description: square blue apples on a tree with circular yellow leaves
Entities: apples, tree, leaves
Activities:
Colors: blue, yellow
Counting:
Text:
Other attributes: square, circular
Questions and programs are below:
About apples (food):
objectEval(image, 'apple')
vqa(image, 'are there apples on the tree?', 'yes,no', 'yes')
vqa(image, 'what type of fruit is on the tree?', 'apples,oranges,bananas,pears', 'apples')
objectEval(image, 'tree')
vqa(image, 'what type of plant is this?', 'tree,flower,bush,grass', 'tree')
About leaves (object):
vqa(image, 'are there leaves on the tree?', 'yes,no', 'yes')
vqa(image, 'what is on the tree?', 'leaves,branches,roots,flowers', 'leaves')
About blue (color):
vqa(image, 'are the apples blue?', 'yes,no', 'yes')
vqa(image, 'what color are the apples?', 'blue,red,yellow,green', 'blue')
About yellow (color):
vqa(image, 'are the leaves yellow?', 'yes,no', 'yes')
vqa(image, 'what color are the leaves?', 'yellow,red,blue,green', 'yellow')
About square (shape):
vqa(image, 'are the apples square?', 'yes,no', 'yes')
vqa(image, 'what shape are the apples?', 'square,round,oval,triangle', 'square')
About circular (shape):
vqa(image, 'are the leaves circular?', 'yes,no', 'yes')
vqa(image, 'what shape are the leaves?', 'circular,square,oval,triangle', 'circular')

Description: Overview of a pot of vegetable soup with wooden spoon on a stainless steel stove top.
Entities: pot, vegetable soup, spoon, stove top
Activities:
Colors:
Counting:
Text:
Other attributes: overview, wooden, stainless steel
Questions and programs are below:
About pot (object):
objectEval(image, 'pot')
vqa(image, 'what is in the pot?', 'vegetable soup,chicken soup,beef soup,fish soup', 'vegetable soup')
vqa(image, 'what type of container is this?', 'pot,bowl,cup,glass', 'pot')
About vegetable soup (food):
objectEval(image, 'vegetable soup')
vqa(image, 'what type of food is this?', 'vegetable soup,chicken soup,beef soup,fish soup', 'vegetable soup')
About spoon (object):
objectEval(image, 'spoon')
vqa(image, 'what type of utensil is this?', 'spoon,fork,knife,chopsticks', 'spoon')
About stove top (object):
objectEval(image, 'stove top')
vqa(image, 'what type of appliance is this?', 'stove top,oven,microwave,refrigerator', 'stove top')
About overview (attribute):
vqa(image, 'is this an overview?', 'yes,no', 'yes')
About wooden (material):
vqa(image, 'is the spoon wooden?', 'yes,no', 'yes')
vqa(image, 'what is the material of the spoon?', 'wooden,metal,plastic,glass', 'wooden')
About stainless steel (material):
vqa(image, 'is the stove top stainless steel?', 'yes,no', 'yes')
vqa(image, 'what is the material of the stove top?', 'stainless steel,iron,aluminum,copper', 'stainless steel')

Description: """

def parse_resp(resp):
    resp = resp.split('\n')
    
    modules = [ ]
    
    for line_number in range(7, len(resp)):
        line = resp[line_number]
        if not line.startswith('About '):
            modules.append(line)

    return modules

def get_question_and_answers(data, use_azure=False, engine="gpt-3.5-turbo"):
    caption = data['caption'] + ""

    this_prompt = prompt + caption + "\"\nEntities"
    resp = openai_completion(this_prompt, use_azure, engine)
    
    program = parse_resp(resp)
    data["output"] = ';'.join(program)

    return data

def openai_completion(prompt, use_azure=False, engine="gpt-3.5-turbo"):
    if use_azure:
        resp = openai.ChatCompletion.create(
            engine=engine,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
    else:
        resp = openai.ChatCompletion.create(
            model=engine,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

    return resp['choices'][0]['message']['content']


if __name__ == "__main__":
    with open('prompt_generation_input.json', 'r') as f:
        test = json.load(f)
    
    with Pool(3) as p:
        results = list(tqdm(p.imap(get_question_and_answers, test), total=len(test)))

    with open('output.json', 'w') as f:
        json.dump(results, f, indent=2)