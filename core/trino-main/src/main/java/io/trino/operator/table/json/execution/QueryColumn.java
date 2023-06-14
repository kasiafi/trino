/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.trino.operator.table.json.execution;

import com.fasterxml.jackson.databind.JsonNode;
import io.trino.spi.Page;

import java.lang.invoke.MethodHandle;

import static java.util.Objects.requireNonNull;

public class QueryColumn
    implements Column
{
    private final int outputIndex;
    private final MethodHandle methodHandle;

    public QueryColumn(int outputIndex, MethodHandle methodHandle)
    {
        this.outputIndex = outputIndex;
        this.methodHandle = requireNonNull(methodHandle, "methodHandle is null");
    }

    // in LocalExecutionPlanner:
    // get method handle for the function (specialize)
    // bind all arguments that are constant: path, behaviors, parametersRow (empty)
    // FIND A WAY to instantiate the JsonPathInvocationContext per driver and reuse across rows for each function (column)
    // also, Session must be passed

    @Override
    public Object evaluate(long sequentialNumber, JsonNode item, Page input, long position)
    {
        // call the handle for (item)
        throw new UnsupportedOperationException();
    }

    @Override
    public int getOutputIndex()
    {
        return outputIndex;
    }
}
