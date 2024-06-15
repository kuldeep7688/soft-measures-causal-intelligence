# The goals of the Data-Labeling GUI
____
This GUI seeks to optimize the data labeling process of causal relations for fine-tuning LLMs. The GUI seeks to achieve this by minimizing the amount of interactions/actions necessary to create the training data.

## Installing the GUI/webapp
____
Refer to Install.MD and Requirements.txt for installation instructions.

## Inputting an RTF, TXT, or JSON File
### RTF
The ideal RTF would currently be extracting the passages that you want to create relations on. This means removing the headers and only extracting the paragraphs.
An example of the ideal and the original version of an RTF input file is in this directory.
### TXT
A text document should be used as the format when you would like to use more than a sentence at a time as it will read line by line. 
### JSON
A json file should be in the format that the UI outputs, or it will not function properly.
This should be used when you want to edit your data or continue from a previous session.
This format allows for some comparison in the LLM comparison tab, but most of the functionality is not yet added there.

## Functionality of the GUI
____

### Labeling the data step by step

* This UI works by selecting the text within the sentence that you want to assign to a relation.
* When you first open the UI, upload a single file at a time. Ideally, right now only label one paper at a time.
* Once the file(s) is loaded, the UI will ask for some meta-data to add to the paper. The data-labeling UI for the elo comparison will also ask you to enter your name.
* Once the complete button for the metadata has been pressed, the sentences will appear on the bottom of the page and you can hit the "Next" button.
* At this point, the UI will display the first sentence in the paper and you can begin labeling your data.
* Select the text that you want to identify as the "source" or "target" and click the corresponding button, then do the same to label the other. 
  * Click Increase or Decrease to set the direction. If you have another relation in that sentence, hit save relation. 
  * When you hit save relation, it will not reset the current selections, and you cannot save the same relation twice. 
* If you only have one relation in a sentence, you do not have to hit save relation, as the next or back button will save the currently labeled relation, given that it is not a repeat relation.
* When you are finished adding relations to the sentence, click next.
* Even at the final sentence of the paper when you are finished labeling relations, make sure to hit save relation or next/back to save that relation.
* When you are finished hit the download JSON button. This will reset the relation storage and remove the inputted papers.

### Features
#### Edit created relations
* To edit, you must double click a field so that it highlights, then you can edit the text, but you have to hit enter to make any changes. 
** You cannot leave fields blank or they will repopulate with what was there before
* Direction will ONLY accept "increase" or "decrease" 
** Direction will accept also + or - and fill in the table with the "Increase" or "Decrease"
#### Hotkeys
* Shift+s will save the current relation if it is not a duplicate
* s will set the currently selected text to the source
* t will set the currently selected text to the target
* + and - on the keypad will set the direction to increase or decrease
* The right arrow key will move you forward through the texts
* The left arrow key will move you backward through the texts
* Up arrow will set the direction of the relation to increase
* Down arrow will set the direction of the relation to decrease

### Input file quirks

* You can input multiple files, but the output name will be based upon the last file and the day it was saved.
* 
* If you try to input the same file consecutively (input the same file twice back to back) it will not add the text again. However, if you add a different file after it will add that file, then you can add the other document again. I do not know when you would need to do this, but this is an option.
* The other option is to refresh the page and insert the same paper again immediately after.

### Executing program

* If you are launching the program from the terminal, it will give you an IP address for a site (hosted locally) which may be hard to notice initially: 127.0.0.1:8050 

## Version History
____
* 0.1
    * Initial Release
* 0.2
    * Back button, discard button, and inverse relations added.
* 1.0
    * For now, the rest of the owl
