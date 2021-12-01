//
// Created by Thoh Testarossa on 2019-04-05.
//

#pragma once

#ifndef GRAPH_ALGO_TISEXTENDED_HPP
#define GRAPH_ALGO_TISEXTENDED_HPP

template<typename T, typename TBase>
class TIsExtended
{
public:
    static int t(TBase* base){
        return 1;
    }
    static char t(void* t2){
        return 2;
    }
    enum{
        Result = (sizeof(int) == sizeof(t((T*)NULL))),
    };
};

#endif //GRAPH_ALGO_TISEXTENDED_HPP
