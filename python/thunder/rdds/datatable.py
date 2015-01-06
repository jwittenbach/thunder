class RDDTable(object):
    """
    Class for storing and manipulating a table of data backed up by an rdd 
    """

    def __init__(self, rdd):
        # TODO add type/format checking
        self._table = rdd.map(lambda x:list(x))

    def joinTable(self, table, idx1, idx2):
        """
        merge another table into the current table, aligning based on given columns
        """
        # TODO add type/format checking 

        # put tables in (key,value) format with the keys being the columns used for alignment
        rdd1 = self._moveIdxOut(self._table, idx1)
        rdd2 = self._moveIdxOut(table._table, idx2)
        # join the tables by key
        joined = self._flattenJoined(rdd1.join(rdd2))
        # put the joined key back in its original place within the table
        return RDDTable(self._moveIdxIn(joined, idx1))

    def addColumn(self, rdd, idx):
        """
        add a single column to the table from an rdd, aligning based on the key of the rdd and a given index in the table
        """
        # TODO add type/format checking
        return self.joinTable(RDDTable(rdd),idx,0)

    def removeColumn(self, idx):
        """
        remove a column specificied by an index from the table
        """
        if type(idx) is int:
            idx = [idx]

        return RDDTable(self._table.map(lambda x: [x[i] for i in range(len(x)) if not i in idx]))

    def applyFunction(self, function, idx, replace=True):
        """
        apply a function to a column
        """
        if replace:
            return RDDTable(self._table.map(lambda x: x[:idx]+[function(x[idx])]+x[idx+1:]))
        else:
            return RDDTable(self._table.map(lambda x: x+[function(x[idx])]))

    @staticmethod
    def _moveIdxOut(rdd, idx):
        return rdd.map(lambda x: (x[idx], x[:idx]+x[idx+1:]))

    @staticmethod
    def _flattenJoined(rdd):
        return rdd.map(lambda (k, v): (k, v[0]+v[1]))
        
    @staticmethod
    def _moveIdxIn(rdd, idx):
        return rdd.map(lambda (k, v): v[:idx]+[k]+v[idx:])

