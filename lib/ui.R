library(shinyBS)

shinyUI(navbarPage(id="navbar",title="Drive and Arrive - Distracted Driver Detection",
                   theme="grey.css",
                   tabPanel("Introduction",
                            fluidPage( 
                                fluidRow(
                                    column(3, h4('1. Motivation')),
                                    column(3, h4('2. Dataset')),
                                    column(3, h4('3. Algorithms')),
                                    column(3, h4('4. Interface'))
                                        ),
                                fluidRow(
                                  column(3, 
                                         h5('   According to the CDC motor vehicle safety division, one in five car accidents is caused by driver distraction. This translates to 425,000 people injured and 3,000 people killed by distracted driving every year. '),
                                         h5('   This tool can help us detected the distracted driving behavior in time and potentially prevent the accidents from happening.')
                                         ),
                                  column(3, 
                                         h5('   The dataset was collected from a Kaggle competition: State Farm Distracted Driver Detection. It includes two files: imgs.zip, a zipped folder of all (train/test) images, and driver_imgs_list.csv, a list of training images, their subject (driver) id, and class id.')),
                                  column(3, 
                                         h5('   Step 1 - face recognition: OpenCV Haar Cascade Classifier'),
                                         h5('   Step 2 - graph features extraction: layer fc7/fc8/prob from deep feature (Caffe) + OpenCV SIFT features'),
                                         h5('   Step 3 - feature decomposition: PCA. 95% variation remained'),
                                         h5('   Step 4 - RF modeling n=10 (cross validation on KNN/RF/NN)'),
                                         h5('   Step 5 - classification on new image')),
                                  column(3,  
                                         h5('   To detect whether a driver is distracted or not, the user needs to go to the classification tab and upload a photo captured by the camera. '),
                                         h5('   Once the photo is uploaded, the model will try to recognize the driver face and therefore trigger the distraction behavior classification. Alert will be displayed.'))
                                ),
                                fluidRow(
                                  column(3, 
                                         img(src='Distracted.jpeg',
                                             width = 220, height=300)
                                  ),
                                  column(3, 
                                         img(src='kaggle2.png',
                                             width = 220, height=300)),
                                  column(3, 
                                         img(src='algorithm.png',
                                             width = 220, height=300)),
                                  column(3, 
                                         img(src='interface.png',
                                             width = 220, height=300))
                                )
                                      )
                   ),
                   
                   tabPanel("Detection",
                            sidebarPanel(fileInput('x_file', 'Choose a picture to upload',
                                                accept = c('image/jpg', '.jpg')),
                                         imageOutput('x_image',width = 80), 
                                         textOutput('x_classify_progress')),
                            mainPanel( 
                                       h3('Face Recognition'),
                                       imageOutput('x_face', height=300),
                                       h3('Distraction Classification'),
                                       tableOutput('x_prediction'),
                                       textOutput("exampleOutput"),
                                       bsAlert("alert")
                                     )
                            ),
                            
                  
                   tabPanel("Contact",
                            titlePanel(h2("Contact")),
                            mainPanel(tabPanel("Contact",includeMarkdown("contact.md"))
                            ))))

