//
// Created by Thoh Testarossa on 2019-03-08.
//

#pragma once

#ifndef GRAPH_ALGO_MESSAGESET_H
#define GRAPH_ALGO_MESSAGESET_H

#include "../include/deps.h"

#define INVALID_MASSAGE INT32_MAX

template <typename MessageValueType>
class Message
{
public:
    Message(const MessageValueType value);
    Message(int src, int dst, const MessageValueType& value);

    int src;
    int dst;
    MessageValueType value;
};

template <typename MessageValueType>
class MessageSet
{
public:
    MessageSet();
    void insertMsg(const Message<MessageValueType>& m);
    void insertMsgCopy(const Message<MessageValueType> m);

    std::vector<Message<MessageValueType>> mSet;
};

#endif //GRAPH_ALGO_MESSAGESET_H
