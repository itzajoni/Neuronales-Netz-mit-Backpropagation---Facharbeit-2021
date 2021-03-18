/**
 * Copyright (c) 2021 Jonathan Dreisvogt, jdreisvogt.de
 * The source code(summarized as "Software") is property of jdreisvogt.de (Jonathan Dreisvogt)
 * The content of our software is subject to German copyright law.
 * You are allowed to copy, publish and use this software as long as you mention source and copyright
 */

package de.jdreisvogt.sandbox.neuralnet.test;

import java.sql.*;

/**
 * Diese Klasse dient als Interface mit der Datenbank, welche den MNIST Datensatz beinhaltet
 * Es wird ab dem Aufruf des Konstruktors damit gerechnet, dass auf dem System eine Datenbank läuft, die der unter
 * /code/Datenbank-Erstellen/datenbank.sql gespeicherte entspricht, welche unter folgenden Parametern verfügbar ist:
 * Name: 'Facharbeit', Port: 3306, SQL-Dialekt: MYSQL, Benutzername: 'root', Passwort: ''
 */
public class MnistLoader
{
    /**
     * Repräsentiert einen einzelnen Datensatz
     */
    public static class DataSet
    {
        final public int id;

        final public double[] y = new double[10];
        final public double[] x = new double[784];
        final public int correctValue;

        public DataSet(int id, int result)
        {
            this.correctValue = result;
            this.id = id;
            for (int i = 0; i < 10; i++)
            {
                if (result == i) this.y[i] = 1;
                else this.y[i] = 0;
            }
        }
    }

    Connection database;
    DataSet[] data;

    /**
     * Konstruktor
     * Stellt die Verbindung mit der Datenbank her
     */
    public MnistLoader()
    {
        try
        {
            this.database = DriverManager.getConnection("jdbc:mysql://localhost:3306/Facharbeit?useSSL=false", "root", "");
        } catch (Exception e)
        {
            System.out.println(e.getMessage());
            System.out.println("Error: cannot access database.");
        }
    }

    /**
     * Lädt die Datensätze aus der Datenbank in das interne Attribut
     */
    public void load()
    {

        try
        {
            Statement statement = this.database.createStatement();
            Statement length_req = this.database.createStatement();
            ResultSet length_res = length_req.executeQuery("SELECT COUNT(*) FROM mnistimgtbl");
            ResultSet result = statement.executeQuery("SELECT * FROM mnistimgtbl");
            length_res.next();
            this.data = new DataSet[length_res.getInt(1)];
            while (result.next())
            {
                DataSet data = new DataSet(result.getInt("id"), result.getInt("label"));
                for (int i = 0; i < 784; i++)
                {
                    data.x[i] = result.getInt("Pixel" + (i + 1)) / 255d;
                }
                this.data[data.id - 1] = data;
            }
        } catch (SQLException e)
        {
            System.out.println(e.getMessage());
            System.out.println("Error: cannot load data from database");
        }
    }

    /**
     * Gibt ein Array mit sämtlichen geladenen Datensätzen zurück
     * @return Datensätze
     */
    public DataSet[] getAll()
    {
        return this.data;
    }

    /**
     * Gibt einen bestimmten Abschnitt der Daten zurück
     * @param start_index Punkt, ab den der Datensatz kopiert wird
     * @param end_index Punkt, bis zu dem die Daten kopiert werden
     * @return Datensätze
     */
    public DataSet[] getPart(int start_index, int end_index)
    {
        if (end_index >= this.data.length) end_index = this.data.length-1;
        DataSet[] result = new DataSet[end_index - start_index];
        System.arraycopy(this.data, start_index, result, 0, end_index - start_index);
        return result;
    }
}
