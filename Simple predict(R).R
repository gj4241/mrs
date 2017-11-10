library(data.table)
library(dplyr)
setwd("C:\\Users\\gyujin\\Desktop\\kaggle data")

train = fread("train.csv",header = T, sep = ",", encoding = "UTF-8")
test = fread("test.csv",header = T, sep = ",", encoding = "UTF-8")
songs = fread("songs.csv",header = T, sep = ",", encoding = "UTF-8")
members = fread("members.csv",header = T, sep = ",", encoding = "UTF-8")
song_extra_info = fread("song_extra_info.csv",header = T, sep = ",", encoding = "UTF-8")

song = merge(song_extra_info, songs, by="song_id")
data1 = merge(train, song, by="song_id", all.x = T)
data = merge(data1, members, by="msno", all.x = T)

write.csv(data, file = "data.csv")
###########################

data = fread("data.csv")


##### train, test
ind = sample(nrow(data), nrow(data)*0.7)
train = data[ind,]
test = data[-ind,]


### 함수 설정
variable = function(x){
  
  x = x
  
  #train
  trainvar1 = train[ , x, with = F]
  trainvar1[is.na(trainvar1),] = 0   
  trainvar1[trainvar1=="",] = 0
  trainvar2 = sapply(trainvar1, function(x) strsplit(x, split = "\\|"))
  trainvar = unlist(trainvar2)
  ntrvar = sapply(trainvar2, function(x) length(x))
  trtarget = rep(train$target, ntrvar)
  trainvar3 = data.table(variable = trainvar, trtarget)
  varmean = trainvar3[, mean(trtarget), by = "variable"]
  
  #test
  testvar1 = test[ , x, with = F]
  testvar1[is.na(testvar1),] = 0
  testvar1[testvar1=="",] = 0
  testvar2 = sapply(testvar1, function(x) strsplit(x, split = "\\|"))
  testvar = unlist(testvar2)
  ntevar = sapply(testvar2, function(x) length(x))
  index = rep(1:nrow(testvar1), ntevar)
  testvar3 = data.table(index, variable = testvar)
  testvar4 = merge(testvar3, varmean, by = "variable", all.x = T)
  testvar4[is.na(V1), V1 := 0.5]
  variable = testvar4[, mean(V1), by = "index"] %>% arrange(index)
  v1 = variable$V1
}

### 변수 변환

source_system_tab = variable("source_system_tab")
source_screen_name = variable("source_screen_name")
source_type = variable("source_type")
song_length = variable("song_length")
genre_ids = variable("genre_ids")
artist_name = variable("artist_name")
composer = variable("composer")
lyricist = variable("lyricist")
language = variable("language")
city = variable("city")
bd = variable("bd")
gender = variable("gender")
registered_via = variable("registered_via")
registration_init_time = variable("registration_init_time")
expiration_date = variable("expiration_date")


### 정형 데이터 
ydata = data.table(source_system_tab, source_screen_name, source_type, genre_ids, artist_name, composer,lyricist,gender)
pred = apply(ydata, 1, mean)
pred1 = ifelse(pred>=0.5, 1, 0)

table(test$target==pred1)
