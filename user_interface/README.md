# The goals of the GUI
____
This GUI seeks to optimize the data labeling process of causal relations for fine-tuning LLMs. The GUI seeks to achieve this by minimizing the amount of interactions/actions necessary to create the training data.

## Installing the GUI/webapp
____
Refer to Install.MD and Requirements.txt for installation instructions.

## Inputting an RTF File

The ideal RTF would currently be extracting the passages that you want to create relations on. This means removing the headers and only extracting the paragraphs.
An example of the ideal and the original version of an RTF input file is in this directory.

## Functionality of the GUI
____

### Labeling the data step by step

* This UI works by selecting the text within the sentence that you want to assign to a relation.
* When you first open the UI, upload a single file at a time. Ideally, right now only label one paper at a time.
* Once the UI has loaded in the sentences, they will appear on the bottom of the page and you can hit the "Next" button.
* At this point, the UI will display the first sentence in the paper and you can begin labeling your data.
* Select the text that you want to identify as the "source" or "target" and click the corresponding button, then do the same to label the other. 
  * Click Increase or Decrease to set the direction. If you have another relation in that sentence, hit save relation. 
  * When you hit save relation, it will not reset the current selections, and you cannot save the same relation twice. 
* If you only have one relation in a sentence, you do not have to hit save relation, as the next or back button will save the currently labeled relation, given that it is not a repeat relation.
* When you are finished adding relations to the sentence, click next.
* Even at the final sentence of the paper when you are finished labeling relations, make sure to hit save relation or next/back to save that relation.
* When you are finished hit the download JSON button. This will reset the relation storage and remove the inputted papers.

### Input file quirks

* You can input multiple files, but the output name will be based upon the last file and the day it was saved.
* 
* If you try to input the same file consecutively (input the same file twice back to back) it will not add the text again. However, if you add a different file after it will add that file, and you can add the other document again. I do not know when you would need to do this, but this is an option.
* The other option is to refresh the page and insert the same paper again immediately after.

### Executing program

* If you are launching the program from the terminal, it will give you an IP address for a site (hosted locally) which may be hard to notice initially: 127.0.0.1:8050 

## Version History
____
* 0.1
    * Initial Release
* 0.2
    * Back button, discard button, and inverse relations added.
