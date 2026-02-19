# Tecan Spark Module
This module can import and process data from experiments recorded by the Tecan Spark plate reader. The Spark plate reader enables users to define their own measurements by chaining different instructions and detection methods together. This makes it powerful, but the data format can become incredibly painful since the layout changes depending on the specific settings of the instrument. This module is an attempt to automatically parse any data correctly with minimal user input.

You can find examples in this [notebook](../../examples/tecan_spark.ipynb).

## Experiment
The experiment class is the main class for reading excel files from Tecan Spark. It automatically parses and stores all meta data (software versions, serial numbers, etc.) and parses the data.
```python
import wetlabtools

result_file = '../../examples/tecan_spark.xlsx'
experiment = wetlabtools.spark.Experiment(result_file)
```

## Actions
To read the settings and data correctly, each measurement (Tecan calls them detection strips) or plate manipulation is implemented as an action. The action class knows how to parse the settings and data and it keeps track of its parents and childs (e.g. an absorbance measurement can be a child of a kinetic loop, which enables users to record time series of absorbance, for example for enzymatic assays). Actions have functions to find their parent or childs. 
```python
from wetlabtools.spark.actions import PlateAction, KineticAction, AbsorbanceAction

action_1 = PlateAction()
action_2 = KineticAction()
action_3 = AbsorbanceAction(label="Absorbance_405nm")

action_1.add_child(action_2)
action_2.add_child(action_3)
action_1.find_descendants(AbsorbanceAction)
action_2.get_parent()
```

## Workflow
The workflow class manages the hierarchy and order of the different actions. It has a couple convenient functions to visualize the order of actions and find actions of specific type.
```python
from wetlabtools.spark.workflow import Workflow

workflow = Workflow()

# add action 1 with all its childs
workflow.add_action(action_1)

absorbance_ations = workflow.find_all(AbsorbanceAction)
print(workflow.to_tree())
```
