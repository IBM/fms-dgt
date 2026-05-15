// Copyright The DiGiT Authors
// SPDX-License-Identifier: Apache-2.0

'use client';

import { isEmpty } from 'lodash';
import { useMemo } from 'react';

import {
  DataTable,
  TableContainer,
  Table,
  TableHead,
  TableRow,
  TableHeader,
  TableBody,
  TableCell,
  CodeSnippet,
} from '@carbon/react';

import { DataPoint } from '@/types/custom';
import { truncate } from '@/src/common/utilities/string';

import classes from './DataPointsTable.module.scss';

function populateTable(
  datapoints: DataPoint[],
): [{ key: string; header: string }[], { id: string; [key: string]: any }[]] {
  const dataPoint = !isEmpty(datapoints) ? datapoints[0] : {};
  const headers = Object.keys(dataPoint).map((key) => ({ key, header: key }));
  const rows: { id: string; [key: string]: any }[] = datapoints.map(
    (dataPoint, dataPointIdx) => ({
      id: `${dataPointIdx + 1}`,
      ...Object.fromEntries(Object.entries(dataPoint)),
    }),
  );
  return [headers, rows];
}

// Renders all rows passed in — pagination is the caller's responsibility.
export default function DataPointsTable({
  datapoints,
}: {
  datapoints: DataPoint[];
}) {
  const [headers, rows] = useMemo(
    () => populateTable(datapoints),
    [datapoints],
  );

  return (
    <>
      {headers && rows ? (
        <>
          <div className={classes.table}>
            <DataTable rows={rows} headers={headers}>
              {({
                rows,
                headers,
                getHeaderProps,
                getRowProps,
                getTableProps,
                getTableContainerProps,
              }) => (
                <TableContainer
                  className={classes.table}
                  {...getTableContainerProps()}
                >
                  <Table {...getTableProps()}>
                    <TableHead>
                      <TableRow>
                        {headers.map((header) => {
                          const { key, ...headerProps } = getHeaderProps({
                            header,
                          });
                          return (
                            <TableHeader
                              id={header.key}
                              key={header.key}
                              {...headerProps}
                            >
                              {header.header}
                            </TableHeader>
                          );
                        })}
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {rows.map((row) => {
                        const { key, ...rowProps } = getRowProps({ row });
                        return (
                          <TableRow key={row.id} {...rowProps}>
                            {row.cells.map((cell) => (
                              <TableCell key={cell.id}>
                                {Array.isArray(cell.value) ||
                                typeof cell.value === 'object' ? (
                                  <CodeSnippet
                                    type="multi"
                                    hideCopyButton
                                    wrapText
                                    maxCollapsedNumberOfRows={10}
                                  >
                                    {JSON.stringify(cell.value, null, 2)}
                                  </CodeSnippet>
                                ) : typeof cell.value === 'boolean' ? (
                                  cell.value ? (
                                    'Yes'
                                  ) : (
                                    'No'
                                  )
                                ) : typeof cell.value === 'string' ? (
                                  truncate(cell.value, 150)
                                ) : (
                                  cell.value
                                )}
                              </TableCell>
                            ))}
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </DataTable>
          </div>
        </>
      ) : null}
    </>
  );
}
