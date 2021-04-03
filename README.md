## Skin Detective Back-End

 **The backend application is stateless. it receives an image, analyse it and returns a response in a form of JSON**

### ***Deployment process:***

1. run the setup.sh script file `path/to/file/setup.sh`

**Pre-Conditions:**

* designed for linux-based operation systems.
* docker installed
* default port is 80 - can be changed according to requirements

### ***Usage***

Description: Analyse mole </br>
Usage: POST /api/analyze?dpi=${dpi} </br>
Consumes: image/png </br>
Produces: application/json </br>
Sample Output: </br>
```json
[{
   "asymmetric_score":1.0,
   "border_score":0.2529822128134704,
   "classification_score":0.4663564443588257,
   "color_score":0.027777777777777776,
   "final_score":0.7144696515795925,
   "mole_center":[
      110,
      126
   ],
   "mole_radius":99.36800289831733,
   "size_score":0.1
}]
```

### ***Artificial Neural Networks***

### **Segmentation network**

**Input**: RGB frame(250x250) </br>
**Output**: GrayScale mask(250x250) - every pixel that relates to the skin lesion will be on (value = 255), otherwise off (value = 0)

<img src="app\files\segmentation_output_examples\output_1.jpg">

### ***Activities***

| `Home Activity` | `Camera Activity` |
| :---: | :---: |
| <img src="app/pictures/Home_Activity.jpeg" width="300"> | <img src="app/pictures/Camera_Activity.jpeg" width="300"> |
| `Analyse Results Activity` | `Results Activity` |
|  <img src="app/pictures/Analyse_Results_Activity.jpeg" width="300"> | <img src="app/pictures/Results_Activity.jpeg" width="300"> |

### ***Netron preview***

The model is `quantization aware trained` that was converted by `TOCO converter` to `tflite` format
<img src="app/pictures/Netron_Preview.png">

### ***UML class diagrams***

Activity | SVG file | PNG picture |
| :---: | :---: | :---: |
| Home Activity |[home_activity.svg](app/UML/home_activity.svg) | [home_activity.png](app/UML/home_activity.png) |
| Camera Activity |[camera_activity.svg](app/UML/camera_activity.svg) | [camera_activity.png](app/UML/camera_activity.png) |
| Analyse Results Activity |[analyse_results_activity.svg](app/UML/analyse_results_activity.svg) | [analyse_results_activity.png](app/UML/analyse_results_activity.png) |
| Results Activity |[results_activity.svg](app/UML/results_activity.svg) | [results_activity.png](app/UML/results_activity.png) |
| Database |[database.svg](app/UML/database.svg) | [database.png](app/UML/database.png) |