wd<-"/Users/YaqingXie/Desktop/BDA_Project"
setwd(wd)
library(shinyBS)


shinyServer(function(input, output, session) {
 
  x_values <- reactiveValues(x_tops = NULL, x_similars = NULL, x_sim_genre = NULL,
                             x_progress = NULL, x_predict_table = NULL, x_alarm = NULL)
  
  output$x_image <- renderImage({
    x_fileName <- input$x_file$name
    if(is.null(x_fileName)){
      return(list(src=paste0(wd,'/figs/ask_for_image.png'), contentType="image/png",alt="No Image", width=260))
    }
    else{
      x_fileImage <- paste0(wd , '/test_imgs/', x_fileName)
      return(list(src=x_fileImage, contentType="image/jpg",alt="No Image", width=260))
    }
  }, deleteFile = FALSE)
  
  output$x_face <- renderImage({
    x_fileName <- input$x_file$name
    if(is.null(x_fileName)){
      return(list(src=paste0(wd,'/figs/ask_for_image.png'), contentType="image/png",alt="No Image", width=260))
    }
    else{
      x_fileImage <- paste0(wd , '/output/face_recognition/', substring(x_fileName, 1, nchar(x_fileName)-4), '_face.jpg')
      return(list(src=x_fileImage, contentType="image/jpg",alt="No Image", width=400))
    }
  }, deleteFile = FALSE)

  observeEvent(input$x_file, {
    if(is.integer(x_values$x_predict_table[1,1])){ 
      if(x_values$x_predict_table[1,1] >= 0){
        closeAlert(session, "exampleAlert")
        x_values$x_is_alram <- 0
      }
    }
    x_values$x_progress <- NULL
    x_values$x_predict_table <- NULL
    x_fileName <- input$x_file$name
    system('source ~/.profile')
    system('source ~/.bashrc')
    spark <- '/usr/local/Cellar/spark-2.0.1-bin-hadoop2.7/bin'
    system(paste0(spark, '/spark-submit ', wd, '/lib/face_recognition.py'))
    system(paste0(spark, '/spark-submit ', wd, '/lib/caffe_feature_test.py'))
    #system(paste0(spark, '/spark-submit ', wd, '/lib/gradient_feature_test.py'))
    system(paste0(spark, '/spark-submit ', wd, '/lib/classification_model_test.py ', x_fileName))
    library(data.table)
    x_values$x_progress <- 'Done!'
    x_fileName <- paste0('./output/prediction/', x_fileName, '_prediction.csv')
    x_pre_label <- read.csv(x_fileName)
    x_values$x_predict_table <- x_pre_label
    x_values$x_alarm <- x_pre_label[1,1]
  })
  
  observe({
    if(is.integer(x_values$x_predict_table[1,1])){ 
      if(x_values$x_predict_table[1,1] > 0)
      {
        showModal(modalDialog(
          title = "PLEASE DRIVE SAFE!",
          paste0('It\'s not safe ', x_values$x_predict_table[1,2],'!'),
          easyClose = TRUE,
          footer = NULL
        ))
        createAlert(session, "alert", "exampleAlert", title = "PLEASE DRIVE SAFE!", style = 'danger',
                    content = paste0('It\'s not safe ', x_values$x_predict_table[1,2],'!'), append = FALSE)
      }
      else{
        createAlert(session, "alert", "exampleAlert", title = "Drive and arrive :)",
                    content = paste0('Please keep safe driving <3'), append = FALSE)
      }
    }
  })
  
  output$x_classify_progress <- renderText({
    return(x_values$x_progress)
  })
  
  output$x_prediction <- renderTable({
    return(x_values$x_predict_table)
  })
  
})