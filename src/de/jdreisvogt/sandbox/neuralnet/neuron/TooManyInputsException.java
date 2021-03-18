/**
 * Copyright (c) 2021 Jonathan Dreisvogt, jdreisvogt.de
 * The source code(summarized as "Software") is property of jdreisvogt.de (Jonathan Dreisvogt)
 * The content of our software is subject to German copyright law.
 * You are allowed to copy, publish and use this software as long as you mention source and copyright
 */

package de.jdreisvogt.sandbox.neuralnet.neuron;

/**
 * Stellt eine Fehlermeldung dar, sollten zu viele Eingaben an ein Neuron übergeben werden
 */
public class TooManyInputsException extends Exception
{
    int layerId;
    int neuronId;
    int inputLength;
    int weightLength;

    public TooManyInputsException(int layerId, int neuronId, int inputLength, int weightLength)
    {
        this.inputLength = inputLength;
        this.weightLength = weightLength;
        this.layerId = layerId;
        this.neuronId = neuronId;
    }

    @Override
    public String toString()
    {
        return "TooManyInputsException{" +
                "layerId=" + layerId +
                ", neuronId=" + neuronId +
                "}: There are fewer weights than inputs. (" + this.inputLength + " Inputs and " + this.weightLength + " weights.";
    }
}
