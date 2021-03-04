# Search algorithm test
## Search right
### Search description
The search on the right is based on finding the topic or point of interest in the dictionary of words, obtaining the initial point of the word in coordinates, followed by giving a search margin down and to the right detecting the end based on the search topics.


    *Name test: 1.1.1
    *File input: LRT001.pdf
    *Point of interest:Insured
    *Output:D4 Inc
    *Status: 
    
    *Name test: 1.1.2
    *File input: LRT001.pdf
    *Point of interest:As of  
    *Output: 02/18/2016  
    *Status: 
    
    *Name test: 1.1.4
    *File input: LRT002.pdf
    *Point of interest:As of 
    *Output:06/03/2015  
    *Status:
    
    *Name test: 1.1.5
    *File input:LRT003.pdf - Page 5 
    *Point of interest:Account 
    *Output:8005109
    *Status:
    
    *Name test: 1.1.6
    *File input: LRT003.pdf - Page 5
    *Point of interest:Total Incurred
    *Output:1,089.46 
    *Status:
    
    *Name test: 1.1.7
    *File input: LRT004.pdf
    *Point of interest:Date Reported as of
    *Output:January 1, 2020 
    *Status:
    
    *Name test: 1.1.8
    *File input: LRT004.pdf
    *Point of interest: Insured
    *Output:Mel’s Transportation II, LCC
    *Status:
    
    *Name test: 1.1.9
    *File input: LRT004.pdf
    *Point of interest:Policy Number
    *Output:001000022377118
    *Status:
    
    *Name test: 1.1.10
    *File input: LRT004.pdf
    *Point of interest:Status
    *Output:Closed
    *Status:
    
## Search down
### Search description
In the search down, the point of interest is searched in the dictionary by obtaining its coordinates, a margin is given to the right and left of the word and when the page ends .

    *Name test: 1.2.1
    *File input: LRT001.pdf
    *Point of interest:Status
    *Output:Closed, Closed, Closed
    *Status:
    
    *Name test: 1.2.2 
    *File input: LRT001.pdf
    *Point of interest:Incident
    *Output:09/14/2012, 09/14/2012, 05/26/2011
    *Status:
    
    *Name test: 1.2.3 
    *File input: LRT001.pdf
    *Point of interest:Claim Made 
    *Output:09/14/2012, 09/14/2012, 05/26/2011
    *Status:
    
    *Name test: 1.2.4
    *File input: LRT001.pdf
    *Point of interest: Closed
    *Output:02/25/2013, 05/21/2013, 05/21/2012
    *Status:
    
    *Name test: 1.2.5
    *File input: LRT002.pdf
    *Point of interest:Claim
    *Output:2013495999
    *Status:
    
    *Name test: 1.2.6
    *File input: LRT003.pdf - Page 5
    *Point of interest:Claim Number 
    *Output:PHNH14040799831, PHNH14040799996, PHNH14040799996
    *Status:
    
    *Name test: 1.2.7
    *File input: LRT003.pdf - Page 5
    *Point of interest:Status
    *Output:CL, CL, CL
    *Status:
    
    *Name test: 1.2.8
    *File input: LRT003.pdf - Page 5
    *Point of interest:Loss Date
    *Output:04/08/2014, 04/08/2014,04/08/2014
    *Status:
    
    *Name test: 1.2.9
    *File input: LRT003.pdf - Page 5
    *Point of interest:Loss/Exps Paid
    *Output:500.89, 0.00, 588.57
    *Status:
## Search intersection
### Search description
In the intersection search, the point of interest is searched and the search is applied downwards, after the search for the next point of interest is carried out and the search is carried out to the right and where the coordinates intersect the corresponding entity is searched

    *Name test: 1.3.3 
    *File input: LRT001.pdf
    *Point of interest:Expense/Paid
    *Output:0.00, 0.00, 0.00
    *Status:
    
    *Name test: 1.3.4 
    *File input: LRT001.pdf
    *Point of interest:Expense/Oustanding 
    *Output:0.00, 0.00, 0.00
    *Status:
    
    *Name test: 1.3.5
    *File input: LRT001.pdf
    *Point of interest:Total/Incurred 
    *Output:0.00, 0.00, 0.00
    *Status:
    
    *Name test: 1.3.6 
    *File input: LRT002.pdf
    *Point of interest:Indemnity/Paid
    *Output:0.00
    *Status:
    
    *Name test: 1.3.7   
    *File input: LRT002.pdf
    *Point of interest:Indemnity/Outstanding 
    *Output:20,000.00
    *Status:
    
    *Name test: 1.3.8
    *File input: LRT002.pdf
    *Point of interest:Expense/Paid 
    *Output:: 0.00
    *Status:
    
    *Name test: 1.3.9 
    *File input: LRT002.pdf
    *Point of interest:Expense/Oustanding 
    *Output:0.00
    *Status:
    
    *Name test: 1.3.10
    *File input: LRT002.pdf
    *Point of interest:Total/Incurred 
    *Output:20,000.00
    *Status:
    
    *Name test: 1.3.11
    *File input: LRT002.pdf 
    *Point of interest:TOTAL/Incurred 
    *Output:20,000.00
    *Status:
    
    *Name test: 1.3.12
    *File input: LRT004.pdf
    *Point of interest:Paid to Date/Indemnity 
    *Output:0.00
    *Status:
    
    *Name test: 1.3.13
    *File input: LRT004.pdf
    *Point of interest:O/SReserves/Indemnity 
    *Output:0.00
    *Status:
    
    *Name test: 1.3.14
    *File input: LRT004.pdf
    *Point of interest:Paid to Date/Expense 
    *Output:652.07
    *Status:
    
    *Name test: 1.3.15
    *File input: LRT004.pdf
    *Point of interest:O/SReserves/Expense  
    *Output:0.00
    *Status:
    
    *Name test: 1.3.16
    *File input: LRT004.pdf
    *Point of interest:Tot Incurred/Total 
    *Output:745.82
    *Status:
    