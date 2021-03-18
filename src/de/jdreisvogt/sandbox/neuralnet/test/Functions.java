/**
 * Copyright (c) 2021 Jonathan Dreisvogt, jdreisvogt.de
 * The source code(summarized as "Software") is property of jdreisvogt.de (Jonathan Dreisvogt)
 * The content of our software is subject to German copyright law.
 * You are allowed to copy, publish and use this software as long as you mention source and copyright
 */

package de.jdreisvogt.sandbox.neuralnet.test;

import de.jdreisvogt.sandbox.neuralnet.layer.Layer;

/**
 * Sammlung an Funktionen, die auf den KNN ausprobiert wurden
 * Die Facharbeit bezieht sich dabei immer auf die Sigmoidfunktion
 */
public enum Functions
{
    LINEAR(new Layer.Function()
    {
        @Override
        public double compute (double x)
        {
            return x;
        }

        @Override
        public double computeDerivative (double x)
        {
            return 1;
        }

        @Override
        public String getAlgorithmName ()
        {
            return "linear function";
        }
    }),
    SIGMOID(new Layer.Function()
    {
        @Override
        public double compute (double x)
        {
            return 1 / (1 + Math.pow(Math.E, -x));
        }

        @Override
        public double computeDerivative (double x)
        {
            return (this.compute(x) * (1 - this.compute(x)));
        }

        @Override
        public String getAlgorithmName ()
        {
            return "sigmoid function";
        }
    }),

    X3(new Layer.Function()
    {
        @Override
        public double compute (double x)
        {
            return x * x * x;
        }

        @Override
        public double computeDerivative (double x)
        {
            return x * x + 200;
        }

        @Override
        public String getAlgorithmName ()
        {
            return "x hoch 3";
        }
    }),

    STEP(new Layer.Function()
    {
        @Override
        public double compute (double x)
        {
            return x >= 0 ? 1 : 0;
        }

        @Override
        public double computeDerivative (double x)
        {
            return 2; //mit offset 2, da sonst nicht für Backpropagation geeignet
        }

        @Override
        public String getAlgorithmName ()
        {
            return "Step-Function";
        }
    }),

    RELU(new Layer.Function()
    {
        @Override
        public double compute (double x)
        {
            return x > 0 ? x : 0;
        }

        @Override
        public double computeDerivative (double x)
        {
            return x > 0 ? 1 : 0;
        }

        @Override
        public String getAlgorithmName ()
        {
            return "ReLu function";
        }
    }),
    LEAKY_RELU(new Layer.Function()
    {
        @Override
        public double compute (double x)
        {
            return x > 0 ? x : x * 0.01;
        }

        @Override
        public double computeDerivative (double x)
        {
            return 2;//(x > 0 ? 1 : 0.01) + 2;
        }

        @Override
        public String getAlgorithmName ()
        {
            return "leaky ReLu function";
        }
    });


    Functions (Layer.Function function)
    {
        this.function = function;
    }

    public Layer.Function function;

    public static Functions getByAlgorithmName (String name)
    {
        for (Functions function : values())
        {
            if (function.function.getAlgorithmName().equals(name)) return function;
        }
        return null;
    }
}
