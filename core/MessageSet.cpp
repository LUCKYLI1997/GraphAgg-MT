//
// Created by Thoh Testarossa on 2019-03-08.
//

#include "../core/MessageSet.h"

#include <iostream>

template <typename MessageValueType>
Message<MessageValueType>::Message(const MessageValueType value)
{
   this->src = 0;
   this->dst = 0;
   this->value = value;
}

template <typename MessageValueType>
Message<MessageValueType>::Message(int src, int dst, const MessageValueType& value)
{
    this->src = src;
    this->dst = dst;
    this->value = value;
}

template <typename MessageValueType>
MessageSet<MessageValueType>::MessageSet()
{
    this->mSet = std::vector<Message<MessageValueType>>();
}

template <typename MessageValueType>
void MessageSet<MessageValueType>::insertMsg(const Message<MessageValueType>& m)
{
    this->mSet.push_back(m);
}

template <typename MessageValueType>
void MessageSet<MessageValueType>::insertMsgCopy(const Message<MessageValueType> m)
{
    this->mSet.push_back(m);
}
