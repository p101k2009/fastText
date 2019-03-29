/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iomanip>
#include <iostream>
#include <sstream>
#include <queue>
#include <stdexcept>
#include "args.h"
#include "fasttext.h"
#include "com_nuance_FastText.h"
//using namespace fasttext;

void printResult(
    const std::vector<std::pair<fasttext::real, std::string>>& predictions,
    bool printProb,
    bool multiline) {
  bool first = true;
  for (const auto& prediction : predictions) {
    if (!first && !multiline) {
      std::cout << " ";
    }
    first = false;
    std::cout << prediction.second;
    if (printProb) {
      std::cout << " " << prediction.first;
    }
    if (multiline) {
      std::cout << std::endl;
    }
  }
  if (!multiline) {
    std::cout << std::endl;
  }
}

void convertToString(
    const std::vector<std::pair<fasttext::real, std::string>>& predictions, std::stringstream& output) {
  bool first = true;
  for (const auto& prediction : predictions) {
  	if (!first)
  	{
  		output << ";";
  	}

    output << prediction.second;
    output << ":";
    output << prediction.first;

    first = false;
  }
}

//JNIEXPORT jobject JNICALL Java_com_nuance_FastText_predictLine
//  (JNIEnv * env, jobject object, jstring text, jstring model, jint k, jfloat threshold)
//{
//    fasttext::FastText fasttext;
//	char* modelFilePath = (char*)env->GetStringUTFChars(model, NULL);
//    fasttext.loadModel(modelFilePath);
//	char* word = (char*)env->GetStringUTFChars(text, NULL);
//
//	std::vector<std::pair<fasttext::real, std::string>> predictions;
//
//    fasttext.predictLine(word, predictions, k, threshold);
//	printResult(predictions, true, false);
//	env->ReleaseStringUTFChars(model, modelFilePath);
//	env->ReleaseStringUTFChars(text, word);
//	return object;
//}

JNIEXPORT jstring JNICALL Java_com_nuance_FastText_predictLine
  (JNIEnv * env, jobject object, jstring text, jlong model, jint k, jfloat threshold)
{
	fasttext::FastText* fasttext = (fasttext::FastText *) model;
	char* word = (char*)env->GetStringUTFChars(text, NULL);
	std::vector<std::pair<fasttext::real, std::string>> predictions;

	fasttext->predictLine(word, predictions, k, threshold);
//	printResult(predictions, true, false);

	std::stringstream result;
    convertToString(predictions, result);
	env->ReleaseStringUTFChars(text, word);
	return env->NewStringUTF(result.str().c_str());
}

JNIEXPORT jlong JNICALL Java_com_nuance_FastText_loadModel
  (JNIEnv * env, jobject object, jstring model)
{
	fasttext::FastText* fasttext = new fasttext::FastText();
	char* modelFilePath = (char*)env->GetStringUTFChars(model, NULL);
    fasttext->loadModel(modelFilePath);
    env->ReleaseStringUTFChars(model, modelFilePath);
    return (jlong)fasttext;
}
